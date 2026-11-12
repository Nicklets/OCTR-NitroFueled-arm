// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Turnip on KGSL (the Adreno kernel driver of Android) waits for a GPU timestamp with
// IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, passing the timeout in milliseconds. A poll (timeout 0)
// reaches KGSL as 0, which KGSL takes as no timeout: the poll blocks until the GPU gets there.
// Mesa polls that way when a submission signals a timeline semaphore (it collects the finished
// points), in vkGetSemaphoreCounterValue (it collects all of them) and in vkGetFenceStatus. On an
// Adreno 730 the frame's second submission waited ~44 ms for the first, the Vulkan recording
// thread with it, and the GPU ran dry ~11 ms a frame.
// Install() points the Vulkan driver's ioctl import at a wrapper that answers those polls by
// reading the retired timestamp. BB_KGSL_POLL=0 leaves the driver alone.
#pragma once

#include <cstdint>

namespace BbKgslPoll {
/// After the Vulkan device exists (the driver is loaded): patches libvulkan_freedreno's imports.
/// A no-op elsewhere than Linux on AArch64, or when no such driver is loaded.
void Install();
/// Polls answered since the last call, and how many of them found the GPU done.
std::uint64_t TakePolls(std::uint64_t* done = nullptr);
} // namespace BbKgslPoll
