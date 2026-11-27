/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#if ALPAKA_LANG_ONEAPI

#    include <catch2/catch_template_test_macros.hpp>
#    include <catch2/catch_test_macros.hpp>
#    include <sycl/sycl.hpp>

#    include <type_traits>
#    include <utility>
#    include <vector>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(std::tuple{api::oneApi}, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("oneAPI native handle link API", "[thirdPartyInteraction][oneApi]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::makeDeviceSpec(cfg);
    auto selector = onHost::makeDeviceSelector(deviceSpec);
    INFO(deviceSpec.getName());
    if(!selector.isAvailable())
        SKIP("No device available for " << deviceSpec.getName());

    auto const deviceKind = deviceSpec.getDeviceKind();
    sycl::device nativeDevice{detail::SYCLDeviceSelector<ALPAKA_TYPEOF(deviceKind)>{}};
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
