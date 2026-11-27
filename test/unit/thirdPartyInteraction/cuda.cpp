/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#if ALPAKA_LANG_CUDA

#    include <catch2/catch_test_macros.hpp>
#    include <cuda_runtime.h>

#    include <cstdint>
#    include <vector>

using namespace alpaka;

namespace
{
    void checkCuda(cudaError_t status, char const* expression)
    {
        INFO(expression << " failed: " << cudaGetErrorString(status));
        REQUIRE(status == cudaSuccess);
    }
} // namespace

TEST_CASE("CUDA native handle link API", "[thirdPartyInteraction][cuda]")
{
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if(status != cudaSuccess || deviceCount == 0)
        SKIP("No CUDA device available");

    uint32_t const nativeDevice = 0;
    checkCuda(cudaSetDevice(static_cast<int>(nativeDevice)), "cudaSetDevice");

    auto selector = onHost::makeDeviceSelector(api::cuda, deviceKind::nvidiaGpu);

    cudaStream_t nativeStream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&nativeStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

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

    checkCuda(cudaStreamDestroy(nativeStream), "cudaStreamDestroy");
}

#endif
