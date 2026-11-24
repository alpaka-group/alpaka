/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#if ALPAKA_LANG_ONEAPI

#    include <catch2/catch_test_macros.hpp>
#    include <sycl/sycl.hpp>

#    include <utility>
#    include <vector>

using namespace alpaka;

TEST_CASE("oneAPI native handle link API", "[thirdPartyInteraction][oneApi]")
{
    auto selector = onHost::makeDeviceSelector(api::oneApi, deviceKind::intelGpu);
    if(!selector.isAvailable())
        SKIP("No oneAPI Intel GPU device available");

    sycl::platform platform{sycl::gpu_selector_v};
    auto devices = platform.get_devices(sycl::info::device_type::gpu);
    if(devices.empty())
        SKIP("No SYCL GPU device available on the selected platform");

    sycl::device nativeDevice = devices[0];
    sycl::context nativeContext{nativeDevice};
    auto nativeDeviceHandle = std::make_pair(nativeDevice, nativeContext);

    sycl::queue nativeQueue{nativeContext, nativeDevice, {sycl::property::queue::in_order{}}};

    {
        auto device = selector.linkDevice(nativeDeviceHandle, true);
        auto linkedDeviceHandle = device.getNativeHandle();
        CHECK(linkedDeviceHandle.first == nativeDevice);
        CHECK(linkedDeviceHandle.second == nativeContext);

        auto queue = device.linkQueue(queueKind::nonBlocking, nativeQueue, false);
        CHECK(queue.getNativeHandle() == nativeQueue);

        constexpr int extent = 100;
        auto deviceBuffer = onHost::alloc<int>(device, extent);
        onHost::iota(queue, 0, deviceBuffer);
        std::vector<int> hostBuffer(extent);
        onHost::memcpy(queue, hostBuffer, deviceBuffer);
        onHost::wait(queue);
        for(int i = 0; i < extent; ++i)
            CHECK(hostBuffer[i] == i);

        device.unlinkQueue(nativeQueue);

        auto blockingQueue = device.linkQueue(queueKind::blocking, nativeQueue, false);
        CHECK(blockingQueue.getQueueKind() == queueKind::blocking);

        auto blockingBuffer = onHost::alloc<int>(device, extent);
        onHost::iota(blockingQueue, 0, blockingBuffer);
        std::vector<int> blockingHost(extent);
        onHost::memcpy(blockingQueue, blockingHost, blockingBuffer);
        for(int i = 0; i < extent; ++i)
            CHECK(blockingHost[i] == i);

        device.unlinkQueue(nativeQueue);
        selector.unlinkDevice(nativeDeviceHandle);
    }
}

#endif
