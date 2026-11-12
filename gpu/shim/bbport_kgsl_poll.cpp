// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: KGSL timestamp polls that do not block (see bbport_kgsl_poll.h).
#include "bbport_kgsl_poll.h"

#include <atomic>

#if defined(__linux__) && defined(__aarch64__)
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <link.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace BbKgslPoll {
namespace {
std::atomic<std::uint64_t> polls{0}, polls_done{0};

#if defined(__linux__) && defined(__aarch64__)
// msm_kgsl.h (Mesa's copy, src/freedreno/vulkan/msm_kgsl.h).
struct WaitTimestampCtx {
    unsigned int context_id;
    unsigned int timestamp;
    unsigned int timeout; // ms; 0 = no timeout
};
struct ReadTimestampCtx {
    unsigned int context_id;
    unsigned int type;
    unsigned int timestamp; // out
};
constexpr unsigned int KgslIocType = 0x09;
constexpr unsigned long WaitTimestampIoctl = _IOW(KgslIocType, 0x7, WaitTimestampCtx);
constexpr unsigned long ReadTimestampIoctl = _IOWR(KgslIocType, 0x16, ReadTimestampCtx);
constexpr unsigned int TimestampRetired = 2; // KGSL_TIMESTAMP_RETIRED

// The driver's ioctl. Its prototype is variadic; on AArch64 Linux the third argument travels in
// the same register either way.
int Ioctl(int fd, unsigned long request, ...) {
    va_list args;
    va_start(args, request);
    void* arg = va_arg(args, void*);
    va_end(args);
    if (request == WaitTimestampIoctl && arg) {
        const auto* wait = static_cast<const WaitTimestampCtx*>(arg);
        if (wait->timeout == 0) {
            ReadTimestampCtx read{wait->context_id, TimestampRetired, 0};
            if (::ioctl(fd, ReadTimestampIoctl, &read) == 0) {
                polls.fetch_add(1, std::memory_order_relaxed);
                // Timestamps wrap: Turnip compares them the same way (timestamp_cmp).
                if (std::int32_t(read.timestamp - wait->timestamp) >= 0) {
                    polls_done.fetch_add(1, std::memory_order_relaxed);
                    return 0;
                }
                errno = ETIMEDOUT; // what Turnip expects from a wait that timed out
                return -1;
            }
        }
    }
    return ::ioctl(fd, request, arg);
}

struct Patch {
    int imports = 0;
    const char* name = nullptr;
};

template <typename T>
T Relocated(ElfW(Addr) value, ElfW(Addr) base) {
    // glibc rewrites these dynamic entries to addresses when it loads the object; a loader that
    // does not leaves offsets from the load address.
    return reinterpret_cast<T>(value < base ? value + base : value);
}

int PatchObject(dl_phdr_info* info, size_t, void* data) {
    auto* patch = static_cast<Patch*>(data);
    if (!info->dlpi_name || !std::strstr(info->dlpi_name, "libvulkan_freedreno")) {
        return 0;
    }
    const ElfW(Addr) base = info->dlpi_addr;
    const ElfW(Dyn)* dyn = nullptr;
    ElfW(Addr) relro_start = 0, relro_end = 0;
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr)& phdr = info->dlpi_phdr[i];
        if (phdr.p_type == PT_DYNAMIC) {
            dyn = reinterpret_cast<const ElfW(Dyn)*>(base + phdr.p_vaddr);
        } else if (phdr.p_type == PT_GNU_RELRO) {
            relro_start = base + phdr.p_vaddr;
            relro_end = relro_start + phdr.p_memsz;
        }
    }
    if (!dyn) {
        return 0;
    }
    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;
    const ElfW(Rela)* tables[2]{};
    size_t sizes[2]{};
    for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; ++d) {
        switch (d->d_tag) {
        case DT_SYMTAB:
            symtab = Relocated<const ElfW(Sym)*>(d->d_un.d_ptr, base);
            break;
        case DT_STRTAB:
            strtab = Relocated<const char*>(d->d_un.d_ptr, base);
            break;
        case DT_JMPREL:
            tables[0] = Relocated<const ElfW(Rela)*>(d->d_un.d_ptr, base);
            break;
        case DT_PLTRELSZ:
            sizes[0] = d->d_un.d_val;
            break;
        case DT_RELA:
            tables[1] = Relocated<const ElfW(Rela)*>(d->d_un.d_ptr, base);
            break;
        case DT_RELASZ:
            sizes[1] = d->d_un.d_val;
            break;
        default:
            break;
        }
    }
    if (!symtab || !strtab) {
        return 0;
    }
    const long page = sysconf(_SC_PAGESIZE);
    for (int t = 0; t < 2; ++t) {
        if (!tables[t]) {
            continue;
        }
        for (size_t i = 0; i < sizes[t] / sizeof(ElfW(Rela)); ++i) {
            const ElfW(Rela)& rela = tables[t][i];
            const auto type = ELF64_R_TYPE(rela.r_info);
            if (type != R_AARCH64_JUMP_SLOT && type != R_AARCH64_GLOB_DAT) {
                continue;
            }
            const ElfW(Sym)& sym = symtab[ELF64_R_SYM(rela.r_info)];
            if (std::strcmp(strtab + sym.st_name, "ioctl") != 0) {
                continue;
            }
            void** slot = reinterpret_cast<void**>(base + rela.r_offset);
            void* start = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(slot) &
                                                ~std::uintptr_t(page - 1));
            // Imports bound at load time sit in the RELRO range (read-only after loading):
            // writable for the store only. Lazily bound ones (.got.plt) stay writable: the
            // loader writes there when it resolves another function on its first call.
            if (mprotect(start, page, PROT_READ | PROT_WRITE) != 0) {
                continue;
            }
            *slot = reinterpret_cast<void*>(&Ioctl);
            const auto at = reinterpret_cast<ElfW(Addr)>(slot);
            if (at >= relro_start && at < relro_end) {
                mprotect(start, page, PROT_READ);
            }
            ++patch->imports;
        }
    }
    patch->name = info->dlpi_name;
    return 1;
}
#endif
} // namespace

void Install() {
#if defined(__linux__) && defined(__aarch64__)
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    if (const char* env = std::getenv("BB_KGSL_POLL"); env && env[0] == '0') {
        return;
    }
    Patch patch{};
    dl_iterate_phdr(PatchObject, &patch);
    if (patch.name) {
        std::printf("GPU: KGSL timestamp polls answered without waiting (%d ioctl imports of %s; "
                    "BB_KGSL_POLL=0 turns it off)\n",
                    patch.imports, patch.name);
    }
#endif
}

std::uint64_t TakePolls(std::uint64_t* done) {
    if (done) {
        *done = polls_done.exchange(0, std::memory_order_relaxed);
    }
    return polls.exchange(0, std::memory_order_relaxed);
}
} // namespace BbKgslPoll
