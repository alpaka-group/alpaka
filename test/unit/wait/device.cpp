/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

// any value for testing
int const testValue = 49;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

// Simple write kernel used to create actual device work for testing wait(device).
struct WaitTestWriteKernel
{
    template<typename T_Acc, typename T_Buf>
    ALPAKA_FN_ACC void operator()(T_Acc const&, T_Buf out) const
    {
        out[0] = testValue;
    }
};

template<typename T_DataType>
void executeWaitTest(auto& device, auto const& exec)
{
    using DataType = T_DataType;

    constexpr uint32_t num = 3;
    // for multiple concurrent queues
    constexpr uint32_t numQueues = num;
    // for multiple ops at a single queue test
    constexpr uint32_t numOpsPerQueue = num;

    // Test 1: Multiple concurrent queues

    std::vector<decltype(device.makeQueue())> queues;
    // Create host and device buffers
    std::vector<decltype(onHost::alloc<DataType>(device, Vec<uint32_t, 1>{1u}))> devBufs;
    std::vector<decltype(onHost::allocHostLike(onHost::alloc<DataType>(device, Vec<uint32_t, 1>{1u})))> hostBufs;

    // allocate memory and create queues
    constexpr auto extent = Vec<uint32_t, 1>{1u};
    for(uint32_t i = 0; i < numQueues; ++i)
    {
        queues.push_back(device.makeQueue());
        devBufs.push_back(onHost::alloc<DataType>(device, extent));
        hostBufs.push_back(onHost::allocHostLike(devBufs.back()));
    }

    // enqueue work on multiple queues concurrently
    constexpr auto frameSize = Vec<uint32_t, 1>{1u};
    for(uint32_t i = 0; i < numQueues; ++i)
    {
        auto kernelBundle = KernelBundle{WaitTestWriteKernel{}, devBufs[i]};
        queues[i].enqueue(onHost::FrameSpec{extent, frameSize, exec}, kernelBundle);
        // Note: We deliberately don't wait on individual queues here
    }

    // Test 2: Multiple async operations on SAME queue
    auto singleQueue = device.makeQueue();

    // Create host and device buffers
    std::vector<decltype(onHost::alloc<DataType>(device, extent))> singleQueueBufs;
    std::vector<decltype(onHost::allocHostLike(onHost::alloc<DataType>(device, extent)))> singleQueueHostBufs;

    for(uint32_t i = 0; i < numOpsPerQueue; ++i)
    {
        singleQueueBufs.push_back(onHost::alloc<DataType>(device, extent));
        singleQueueHostBufs.push_back(onHost::allocHostLike(singleQueueBufs.back()));

        // Enqueue multiple operations on same queue - they queue up asynchronously
        auto kernelBundle = KernelBundle{WaitTestWriteKernel{}, singleQueueBufs[i]};
        singleQueue.enqueue(onHost::FrameSpec{extent, frameSize, exec}, kernelBundle);
    }

    // TEST: wait for device to complete ALL work on ALL queues (including single queue ops)
    REQUIRE_NOTHROW(onHost::wait(device));

    // copy results of device buffers to host buffers
    for(uint32_t i = 0; i < numQueues; ++i)
    {
        onHost::memcpy(queues[numQueues - i - 1], hostBufs[i], devBufs[i]);
    }
    for(uint32_t i = 0; i < numOpsPerQueue; ++i)
    {
        onHost::memcpy(singleQueue, singleQueueHostBufs[i], singleQueueBufs[i]);
    }

    // TEST: wait for device to complete ALL copies on ALL queues (including single queue ops)
    REQUIRE_NOTHROW(onHost::wait(device));

    // verify results of different queues
    for(uint32_t i = 0; i < numQueues; ++i)
    {
        REQUIRE(hostBufs[i][0] == testValue);
    }

    // verify results of different operations on single queue
    for(uint32_t i = 0; i < numOpsPerQueue; ++i)
    {
        REQUIRE(singleQueueHostBufs[i][0] == testValue);
    }

    UNSCOPED_INFO(
        "wait(device) verified with " << numQueues << " concurrent queues + " << numOpsPerQueue
                                      << " ops on single queue");
}

void prepareAndExecuteWaitTest(auto cfg)
{
    auto deviceSpec = onHost::DeviceSpec{cfg};
    auto exec = alpaka::getExecutor(cfg);

    auto deviceSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!deviceSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    auto deviceCount = deviceSelector.getDeviceCount();
    INFO("device spec: " << deviceSpec.getName());
    INFO("executor   : " << exec.getName());

    // Test wait(device) on all available devices
    for(uint32_t deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx)
    {
        auto device = deviceSelector.makeDevice(deviceIdx);
        INFO("device name: " << device.getName() << " (idx=" << deviceIdx << ")");

        executeWaitTest<uint32_t>(device, exec);
    }
}

TEMPLATE_LIST_TEST_CASE("wait device", "[wait]", TestApis)
{
    auto cfg = TestType::makeDict();
    prepareAndExecuteWaitTest(cfg);
}
