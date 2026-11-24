/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#if ALPAKA_LANG_HIP

#    include <catch2/catch_test_macros.hpp>
#    include <hip/hip_runtime.h>

#    include <cstdint>
#    include <vector>

using namespace alpaka;

namespace
{
    void checkHip(hipError_t status, char const* expression)
    {
        INFO(expression << " failed: " << hipGetErrorString(status));
        REQUIRE(status == hipSuccess);
    }
} // namespace

TEST_CASE("HIP native handle link API", "[thirdPartyInteraction][hip]")
{
    int deviceCount = 0;
    auto status = hipGetDeviceCount(&deviceCount);
    if(status != hipSuccess || deviceCount == 0)
        SKIP("No HIP device available");

    uint32_t const nativeDevice = 0;
    checkHip(hipSetDevice(static_cast<int>(nativeDevice)), "hipSetDevice");

    auto selector = onHost::makeDeviceSelector(api::hip, deviceKind::amdGpu);

    hipStream_t nativeStream = nullptr;
    checkHip(hipStreamCreateWithFlags(&nativeStream, hipStreamNonBlocking), "hipStreamCreateWithFlags");

    {
        auto device = selector.linkDevice(nativeDevice, true);
        CHECK(static_cast<uint32_t>(device.getNativeHandle()) == nativeDevice);

        auto queue = device.linkQueue(queueKind::nonBlocking, nativeStream, false);
        CHECK(queue.getNativeHandle() == nativeStream);

        constexpr int extent = 100;
        auto deviceBuffer = onHost::alloc<int>(device, extent);
        onHost::iota(queue, 0, deviceBuffer);
        std::vector<int> hostBuffer(extent);
        onHost::memcpy(queue, hostBuffer, deviceBuffer);
        onHost::wait(queue);
        for(int i = 0; i < extent; ++i)
            CHECK(hostBuffer[i] == i);

        device.unlinkQueue(nativeStream);

        auto blockingQueue = device.linkQueue(queueKind::blocking, nativeStream, false);
        CHECK(blockingQueue.getQueueKind() == queueKind::blocking);

        auto blockingBuffer = onHost::alloc<int>(device, extent);
        onHost::iota(blockingQueue, 0, blockingBuffer);
        std::vector<int> blockingHost(extent);
        onHost::memcpy(blockingQueue, blockingHost, blockingBuffer);
        for(int i = 0; i < extent; ++i)
            CHECK(blockingHost[i] == i);

        device.unlinkQueue(nativeStream);
        selector.unlinkDevice(nativeDevice);
    }

    checkHip(hipStreamDestroy(nativeStream), "hipStreamDestroy");
}

#endif
