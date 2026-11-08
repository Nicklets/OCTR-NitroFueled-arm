// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the audio mixer thread spends half of its time in one guest loop, scalar float code that
// the JIT translates one instruction at a time. It filters every channel through two cascaded
// biquads and applies a peak-driven gain. On aarch64 that loop runs here, four channels per NEON
// register, with the operations in the guest's order so that the results stay the same.
//
// The hooks sit after the function's coefficient update, on both of its loops, and resume at its
// epilogue. BB_FMOD_NEON=0 leaves the guest code alone; BB_FMOD_NEON=check runs both and compares.
#if defined(__aarch64__)
#include "bbport_fmod_dsp.h"

#include <arm_neon.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace BbFmodDsp {
namespace {
using u64 = std::uint64_t;
using u8 = unsigned char;
constexpr u64 ImageBase = 0x800000000ull;

constexpr u64 LoopSite = 0x16b318f;   // any channel count: test r12d, r12d
constexpr u64 LoopBody = 0x16b3198;   // past its je
constexpr u64 StereoSite = 0x16b343c; // two channels: test r12d, r12d
constexpr u64 StereoBody = 0x16b344a; // past its vmovss to [rbp-0x30] and je
constexpr u64 SaveState = 0x16b3391;  // both loops end here: the peak and envelope are stored
constexpr u64 Epilogue = 0x16b33a1;   // add rsp, 0x48
constexpr u64 Sign = 0x2b10c04;       // the anti-denormal offset, negated every frame
constexpr u8 TestCode[3] = {0x45, 0x85, 0xe4};
constexpr u8 EpilogueCode[4] = {0x48, 0x83, 0xc4, 0x48};

/// Code and constants that have to be there for this loop to be the one written below.
struct Expected {
    u64 offset;
    u8 bytes[16];
    unsigned size;
};
constexpr Expected Signature[] = {
    {0x16b3050, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56}, 8},
    {LoopBody, {0xc5, 0xfa, 0x10, 0x15, 0x64, 0xda, 0x45, 0x01}, 8},
    {StereoBody, {0xc5, 0xfa, 0x10, 0x05, 0xb2, 0xd7, 0x45, 0x01}, 8},
    {0x1db1cc4, {0x00, 0x00, 0x80, 0x3f}, 4},                         // 1.0f
    {0x1dc0540, {0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0x7f}, 8}, // absolute value mask
    {0x1dc0550, {0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x80}, 8}, // sign mask
};

enum Field : std::size_t {
    InputHistory = 0x1f0,
    FirstHistory = 0x270,
    FirstGain = 0x2f0,
    FirstFeedback = 0x2f4,
    FirstInput = 0x2fc,
    FirstFeedforward = 0x300,
    SecondHistory = 0x320,
    SecondGain = 0x3a0,
    SecondFeedback = 0x3a4,
    SecondInput = 0x3ac,
    SecondFeedforward = 0x3b0,
    Current = 0x3b8,
    Previous = 0x3bc,
    Decay = 0x3c0,
    Attack = 0x3c8,
    Release = 0x3d0,
    Threshold = 0x3d8,
    Level = 0x3e0,
    Envelope = 0x3e4,
    Peak = 0x3e8,
    StateSize = 0x3f0,
};
constexpr std::ptrdiff_t HistoryStride = 0x40;

inline float Get(const u8* state, std::size_t field) {
    float value;
    std::memcpy(&value, state + field, sizeof(value));
    return value;
}
inline void Put(u8* state, std::size_t field, float value) {
    std::memcpy(state + field, &value, sizeof(value));
}
inline int GetInt(const u8* state, std::size_t field) {
    int value;
    std::memcpy(&value, state + field, sizeof(value));
    return value;
}
inline float* History(u8* state, std::size_t field, int index) {
    return reinterpret_cast<float*>(state + field + std::ptrdiff_t(index) * HistoryStride);
}

/// The guest loop. Multiply and add stay two operations: a fused one rounds differently.
void Process(u8* state, const float* in, float* out, int frames, int channels, float* sign_slot) {
#pragma clang fp contract(off)
    const float one = 1.0f;
    const float scale = one / Get(state, Threshold);
    const float decay = Get(state, Decay);
    const float attack = Get(state, Attack);
    const float release = Get(state, Release);
    const float level = Get(state, Level);
    float envelope = Get(state, Envelope);
    float peak = Get(state, Peak);

    const float b0 = Get(state, FirstInput), b1 = Get(state, FirstFeedforward);
    const float b2 = Get(state, FirstFeedforward + 4), a1 = Get(state, FirstFeedback);
    const float a2 = Get(state, FirstFeedback + 4), g1 = Get(state, FirstGain);
    const float c0 = Get(state, SecondInput), c1 = Get(state, SecondFeedforward);
    const float c2 = Get(state, SecondFeedforward + 4), d1 = Get(state, SecondFeedback);
    const float d2 = Get(state, SecondFeedback + 4), g2 = Get(state, SecondGain);

    int current = GetInt(state, Current);
    int previous = GetInt(state, Previous);
    float sign = *sign_slot;

    const float32x4_t vb0 = vdupq_n_f32(b0), vb1 = vdupq_n_f32(b1), vb2 = vdupq_n_f32(b2);
    const float32x4_t va1 = vdupq_n_f32(a1), va2 = vdupq_n_f32(a2), vg1 = vdupq_n_f32(g1);
    const float32x4_t vc0 = vdupq_n_f32(c0), vc1 = vdupq_n_f32(c1), vc2 = vdupq_n_f32(c2);
    const float32x4_t vd1 = vdupq_n_f32(d1), vd2 = vdupq_n_f32(d2), vg2 = vdupq_n_f32(g2);

    for (int frame = 0; frame < frames; ++frame) {
        peak = peak - decay;
        float* const input_new = History(state, InputHistory, previous);
        float* const input_old = History(state, InputHistory, current);
        float* const first_new = History(state, FirstHistory, previous);
        float* const first_old = History(state, FirstHistory, current);
        float* const second_new = History(state, SecondHistory, previous);
        float* const second_old = History(state, SecondHistory, current);
        const float32x4_t vsign = vdupq_n_f32(sign);
        int c = 0;
        for (; c + 4 <= channels; c += 4) {
            const float32x4_t x = vaddq_f32(vld1q_f32(in + c), vsign);
            const float32x4_t y_new = vld1q_f32(first_new + c), y_old = vld1q_f32(first_old + c);
            const float32x4_t z_new = vld1q_f32(second_new + c), z_old = vld1q_f32(second_old + c);
            float32x4_t y = vmulq_f32(x, vb0);
            y = vaddq_f32(y, vmulq_f32(vb1, vld1q_f32(input_new + c)));
            y = vaddq_f32(y, vmulq_f32(vb2, vld1q_f32(input_old + c)));
            y = vsubq_f32(y, vmulq_f32(va1, y_new));
            y = vsubq_f32(y, vmulq_f32(va2, y_old));
            y = vmulq_f32(y, vg1);
            float32x4_t z = vmulq_f32(y, vc0);
            z = vaddq_f32(z, vmulq_f32(vc1, y_new));
            z = vaddq_f32(z, vmulq_f32(vc2, y_old));
            z = vsubq_f32(z, vmulq_f32(vd1, z_new));
            z = vsubq_f32(z, vmulq_f32(vd2, z_old));
            z = vmulq_f32(z, vg2);
            vst1q_f32(input_old + c, x);
            vst1q_f32(first_old + c, y);
            vst1q_f32(second_old + c, z);
            // The guest's maxss keeps the running peak against a NaN: maxnm skips them too.
            const float largest = vmaxnmvq_f32(vabsq_f32(z));
            if (largest > peak) {
                peak = largest;
            }
        }
        for (; c < channels; ++c) {
            const float x = in[c] + sign;
            const float y_new = first_new[c], y_old = first_old[c];
            const float z_new = second_new[c], z_old = second_old[c];
            float y = x * b0;
            y = y + b1 * input_new[c];
            y = y + b2 * input_old[c];
            y = y - a1 * y_new;
            y = y - a2 * y_old;
            y = y * g1;
            float z = y * c0;
            z = z + c1 * y_new;
            z = z + c2 * y_old;
            z = z - d1 * z_new;
            z = z - d2 * z_old;
            z = z * g2;
            input_old[c] = x;
            first_old[c] = y;
            second_old[c] = z;
            const float magnitude = std::fabs(z);
            if (magnitude > peak) {
                peak = magnitude;
            }
        }
        const float target = peak * scale;
        const float rate = one < target ? attack : release;
        envelope = target + (envelope - target) * rate;
        const float gain = envelope > one ? level / envelope : level;
        const float32x4_t vgain = vdupq_n_f32(gain);
        c = 0;
        for (; c + 4 <= channels; c += 4) {
            vst1q_f32(out + c, vmulq_f32(vld1q_f32(second_old + c), vgain));
        }
        for (; c < channels; ++c) {
            out[c] = second_old[c] * gain;
        }
        in += channels;
        out += channels;
        std::swap(current, previous);
        sign = -sign;
    }
    if (frames > 0) {
        *sign_slot = sign;
    }
    std::memcpy(state + Current, &current, sizeof(current));
    std::memcpy(state + Previous, &previous, sizeof(previous));
    Put(state, Peak, peak);
    Put(state, Envelope, envelope);
}

float* SignSlot() {
    return reinterpret_cast<float*>(ImageBase + Sign);
}

bool checking = false;

/// BB_FMOD_NEON=check: the native result of the guest loop running now on this thread.
struct Pending {
    bool active = false;
    u64 state = 0, out = 0;
    float sign = 0;
    u8 result[StateSize];
    std::vector<float> samples;
};
thread_local Pending pending;

struct Totals {
    u64 calls = 0, samples = 0, different = 0, state_different = 0, sign_different = 0;
    float largest = 0;
};
thread_local Totals totals;

void Prepare(const GuestRegs& r, int frames, int channels) {
    auto& p = pending;
    const auto* in = reinterpret_cast<const float*>(r.gpr[GUEST_R15]);
    p.active = true;
    p.state = r.gpr[GUEST_RBX];
    p.out = r.gpr[GUEST_R14];
    p.sign = *SignSlot();
    std::memcpy(p.result, reinterpret_cast<const void*>(p.state), StateSize);
    p.samples.resize(std::size_t(frames) * std::size_t(channels));
    Process(p.result, in, p.samples.data(), frames, channels, &p.sign);
}

void Compare() {
    auto& p = pending;
    auto& t = totals;
    p.active = false;
    const auto* guest = reinterpret_cast<const float*>(p.out);
    const auto* state = reinterpret_cast<const u8*>(p.state);
    ++t.calls;
    for (std::size_t i = 0; i < p.samples.size(); ++i) {
        ++t.samples;
        if (std::memcmp(&guest[i], &p.samples[i], sizeof(float)) != 0) {
            ++t.different;
            const float difference = std::fabs(guest[i] - p.samples[i]);
            if (difference > t.largest) {
                t.largest = difference;
            }
        }
    }
    if (std::memcmp(state + InputHistory, p.result + InputHistory, FirstGain - InputHistory) != 0 ||
        std::memcmp(state + SecondHistory, p.result + SecondHistory, SecondGain - SecondHistory) != 0 ||
        std::memcmp(state + Current, p.result + Current, 8) != 0 ||
        std::memcmp(state + Envelope, p.result + Envelope, 8) != 0) {
        ++t.state_different;
    }
    if (std::memcmp(SignSlot(), &p.sign, sizeof(float)) != 0) {
        ++t.sign_different;
    }
    if (t.calls % 2000 == 0) {
        std::printf("FMOD native check: %llu calls, %llu samples, %llu differ (largest %g), "
                    "state differs in %llu calls, offset in %llu\n",
                    (unsigned long long)t.calls, (unsigned long long)t.samples,
                    (unsigned long long)t.different, double(t.largest),
                    (unsigned long long)t.state_different, (unsigned long long)t.sign_different);
        std::fflush(stdout);
    }
}

bool Read(u64 address, void* out, std::size_t size) {
    iovec local{out, size}, remote{reinterpret_cast<void*>(address), size};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == ssize_t(size);
}
} // namespace

bool Handle(GuestRegs& r) {
    const u64 site = r.rip - ImageBase;
    if (site != LoopSite && site != StereoSite && site != Epilogue) {
        return false;
    }
    if (site == Epilogue) {
        if (pending.active && r.gpr[GUEST_RBX] == pending.state) {
            Compare();
        }
        r.gpr[GUEST_RSP] += 0x48;
        r.rip = ImageBase + Epilogue + sizeof(EpilogueCode);
        return true;
    }
    auto* state = reinterpret_cast<u8*>(r.gpr[GUEST_RBX]);
    const int frames = int(std::uint32_t(r.gpr[GUEST_R12]));
    const int channels = int(std::uint32_t(r.gpr[GUEST_R13]));
    if (!checking) {
        Process(state, reinterpret_cast<const float*>(r.gpr[GUEST_R15]),
                reinterpret_cast<float*>(r.gpr[GUEST_R14]), frames, channels, SignSlot());
        r.rip = ImageBase + Epilogue;
        return true;
    }
    if (site == StereoSite) {
        // The instruction after the hooked test: [rbp-0x30] holds the decay in that loop.
        Put(reinterpret_cast<u8*>(r.gpr[GUEST_RBP] - 0x30), 0, Get(state, Decay));
    }
    if (frames == 0) {
        r.rip = ImageBase + SaveState;
        return true;
    }
    Prepare(r, frames, channels);
    r.rip = ImageBase + (site == LoopSite ? LoopBody : StereoBody);
    return true;
}

void Install(Patcher patch) {
    const char* mode = std::getenv("BB_FMOD_NEON");
    if (mode && mode[0] == '0') {
        return;
    }
    checking = mode && !std::strcmp(mode, "check");
    for (const Expected& e : Signature) {
        u8 bytes[16];
        if (!Read(ImageBase + e.offset, bytes, e.size) || std::memcmp(bytes, e.bytes, e.size) != 0) {
            std::printf("FMOD native loop: not this game version, guest code stays (%#llx)\n",
                        (unsigned long long)e.offset);
            return;
        }
    }
    bool ok = patch(ImageBase + LoopSite, TestCode, sizeof(TestCode)) &&
              patch(ImageBase + StereoSite, TestCode, sizeof(TestCode));
    if (ok && checking) {
        ok = patch(ImageBase + Epilogue, EpilogueCode, sizeof(EpilogueCode));
    }
    std::printf("FMOD native loop: %s\n", !ok ? "HOOKS FAILED" : checking ? "comparing with the guest code" : "on");
}
} // namespace BbFmodDsp
#endif
