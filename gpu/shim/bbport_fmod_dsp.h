// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: native code for the audio mixer's hottest guest loop on aarch64 (bbport_fmod_dsp.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

#include "../../src/guest_cpu.h"

namespace BbFmodDsp {
/// Patches one guest instruction: false when the bytes there differ or no hook is free.
using Patcher = bool (*)(std::uint64_t address, const unsigned char* expected, std::size_t size);
#if defined(__aarch64__)
void Install(Patcher patch);
/// True when `regs.rip` is one of these hooks; regs.rip is then where the game resumes.
bool Handle(GuestRegs& regs);
#else
inline void Install(Patcher) {}
inline bool Handle(GuestRegs&) {
    return false;
}
#endif
} // namespace BbFmodDsp
