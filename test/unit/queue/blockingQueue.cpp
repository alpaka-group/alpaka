/* Copyright 2025 Mehmet Yusufoglu, René Widera, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * This test validates the blocking queue functionality with both blocking and non-blocking queue policies.
 * It ensures that blocking queues  synchronize operations while non-blocking queues maintain
 * asynchronous behavior.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <barrier>
#include <thread>
#include <vector>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

// Any value for testing
constexpr uint32_t kTestFillValue = 49u;

// Test kernel for validation
struct BlockingTestKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, uint32_t value) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i.x()] = value;
        }
    }
};

TEMPLATE_LIST_TEST_CASE("blocking queue memory operations", "[bq][memory]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto blockingQueue0 = device.makeQueue(queueKind::blocking);
    auto blockingQueue1 = device.makeQueue(queueKind::blocking);
    auto blockingQueue2 = device.makeQueue(queueKind::blocking);
    constexpr Vec extent = Vec{16u * 1024u * 1024u};
    constexpr uint32_t testValue = kTestFillValue;

    // Test memory allocation and operations with blocking queue
    auto dBuff = onHost::alloc<uint32_t>(device, extent);
    auto hBuff = onHost::allocHostLike(dBuff);

    // Initialize host buffer
    meta::ndLoopIncIdx(extent, [&](auto idx) { hBuff[idx] = 0u; });

    // Test kernel execution with blocking queue 0
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    blockingQueue0.enqueue(
        onHost::FrameSpec{extent / frameSize, frameSize, exec},
        KernelBundle{BlockingTestKernel{}, dBuff, testValue});

    // With blocking queue 1, operations should be synchronous
    // Copy data back - this should also block automatically
    onHost::memcpy(blockingQueue1, hBuff, dBuff);

    // In cases where the blocking queue not works as expected we have a high change of a data race because we would
    // reset the values with an action in another queue while copying the data
    onHost::memset(blockingQueue2, dBuff, 0u);

    // NO explicit wait needed - blocking queue should handle this automatically
    // Verify the data was written correctly
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuff[idx] == testValue); });
}

// -----------------------------------------------------------------------------
// Test validates that a blocking-queue guarantees host-visible completion after each
// enqueue (kernel -> kernel -> memcpy) and produces the final result without calling wait().
// Also exercises allocDeferred under the blocking policy.
// -----------------------------------------------------------------------------
struct WriteValueKernel
{
    uint32_t value;

    ALPAKA_FN_ACC void operator()(auto const& acc, auto view) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{view.getExtents()}))
        {
            view[i.x()] = value;
        }
    }
};

// Simple fill kernel used in event semantics test; defined at namespace scope to satisfy NVCC
struct FillKernel
{
    uint32_t v;

    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i.x()] = v;
        }
    }
};

// enqueue different steps of a compute chain in independent blocking queues
TEMPLATE_LIST_TEST_CASE("blocking queue chained operations", "[bq][chain]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto qBlocking0 = device.makeQueue(queueKind::blocking);
    auto qBlocking1 = device.makeQueue(queueKind::blocking);
    auto qBlocking2 = device.makeQueue(queueKind::blocking);

    // keep one larger extent for coverage
    constexpr Vec extent = Vec{32u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    auto frameSpec = onHost::FrameSpec{extent / frameSize, frameSize, exec};

    // Device buffers
    auto dBufBlocking = onHost::alloc<uint32_t>(device, extent);
    // Host mirrors
    auto hBufBlocking = onHost::allocHostLike(dBufBlocking);

    // 1) Blocking queue: chain two kernels then memcpy, no wait.
    // First kernel writes 3, second overwrites with 11.
    qBlocking0.enqueue(frameSpec, KernelBundle{WriteValueKernel{3u}, dBufBlocking});
    qBlocking1.enqueue(frameSpec, KernelBundle{WriteValueKernel{11u}, dBufBlocking});
    // blocking finished here
    onHost::memcpy(qBlocking2, hBufBlocking, dBufBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBufBlocking[idx] == 11u); });

    // 2) allocDeferred chain on blocking queue: allocate asynchronously, immediately fill then memcpy.
    // should be ready due to blocking policy of the queue
    auto dBufAsync = onHost::allocDeferred<uint32_t>(qBlocking0, extent);
    qBlocking1.enqueue(frameSpec, KernelBundle{WriteValueKernel{kTestFillValue}, dBufAsync});
    auto hBufAsync = onHost::allocHostLike(dBufAsync);
    onHost::memcpy(qBlocking2, hBufAsync, dBufAsync);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBufAsync[idx] == kTestFillValue); });
}

// Test blocking and non-blocking queue should be usable together
TEMPLATE_LIST_TEST_CASE("mixed queues independence", "[bq][mixed]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    constexpr Vec extent = Vec{8u};
    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    auto frameSpec = onHost::FrameSpec{extent / frameSize, frameSize, exec};

    auto qBlocking = device.makeQueue(queueKind::blocking);
    auto qNonBlocking = device.makeQueue(queueKind::nonBlocking);

    // Separate buffers for each queue
    auto dBuf = onHost::alloc<uint32_t>(device, extent);
    auto hBuf = onHost::allocHostLike(dBuf);

    // Blocking queue: immediate completion guaranteed
    qBlocking.enqueue(frameSpec, KernelBundle{WriteValueKernel{3u}, dBuf});
    // Returns after kernel done
    onHost::memcpy(qNonBlocking, hBuf, dBuf);
    onHost::wait(qNonBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuf[idx] == 3u); });
}

TEMPLATE_LIST_TEST_CASE("blocking queue event semantics", "[bq][event]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    // Blocking producer queue, non-blocking consumer queue
    auto qBlocking = device.makeQueue(queueKind::blocking);
    auto qNonBlocking = device.makeQueue(queueKind::nonBlocking);

    // 1. Event recorded on blocking queue is immediately complete
    auto e1 = device.makeEvent();
    // recording
    qBlocking.enqueue(e1);


    // must already be signaled
    CHECK(e1.isComplete());
    // should be a no-op
    onHost::wait(e1);

    // 2. Kernel -> event -> memcpy on blocking queue (no explicit waits)
    constexpr Vec extent = Vec{16u};
    auto dBuf = onHost::alloc<uint32_t>(device, extent);
    auto hBuf = onHost::allocHostLike(dBuf);
    auto hBuf2 = onHost::allocHostLike(dBuf);
    auto hBuf3 = onHost::allocHostLike(dBuf);

    // Use global FillKernel (defined above)

    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    qBlocking.enqueue(onHost::FrameSpec{extent / frameSize, frameSize, exec}, KernelBundle{FillKernel{123u}, dBuf});

    auto e2 = device.makeEvent();
    // all prior work (kernel) complete when this returns
    qBlocking.enqueue(e2);
    CHECK(e2.isComplete());

    // implicit sync via blocking queue
    onHost::memcpy(qBlocking, hBuf, dBuf);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuf[idx] == 123u); });
    // 3. Blocking data generator -> non-blocking consumer/user waits on event
    //   (event already complete; consumer wait should be instantaneous)
    auto e3 = device.makeEvent();
    // enqueue trivial kernel + event on blocking queue
    qBlocking.enqueue(onHost::FrameSpec{extent / frameSize, frameSize, exec}, KernelBundle{FillKernel{77u}, dBuf});
    qBlocking.enqueue(e3);
    CHECK(e3.isComplete());

    // Consumer queue waits on event then copies & validates (must still work)
    qNonBlocking.waitFor(e3);
    onHost::memcpy(qNonBlocking, hBuf2, dBuf);
    onHost::wait(qNonBlocking);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuf2[idx] == 77u); });

    // 4. Non-blocking producer -> blocking consumer not using explicit waits
    auto qNonBlockingProd = device.makeQueue(queueKind::nonBlocking);
    auto qBlockingCons = device.makeQueue(queueKind::blocking);
    auto e4 = device.makeEvent();
    qNonBlockingProd.enqueue(
        onHost::FrameSpec{extent / frameSize, frameSize, exec},
        KernelBundle{FillKernel{55u}, dBuf});
    // may not be complete yet
    qNonBlockingProd.enqueue(e4);
    // NOTE: The non-blocking thread can execute the kernel and the event completion task before
    // the test thread reaches this point (especially on fast CI machines or tiny kernels), making
    // the negative assertion fail. The test only needs to validate that waitFor on the blocking
    // consumer provides the required synchronization and the memcpy observes the produced data.
    if(e4.isComplete())
    {
        INFO("e4 completed early (acceptable: fast execution on non-blocking queue)");
    }
    // non blocking queues are ordered internally but since non-blocking the host, event must be waited.
    // enqueue wait on blocking consumer without explicit wait !!
    qBlockingCons.waitFor(e4);

    // possible data race if the command before is not executed blocking
    onHost::memcpy(qBlockingCons, hBuf3, dBuf);
    // Check for all elements of the extent
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(hBuf3[idx] == 55u); });
}

// blocking queue event-cache behavior tests
TEMPLATE_LIST_TEST_CASE("blocking queue event cache functionality", "[event][blocking-queue]", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());

    SECTION("Rapid event enqueueing")
    {
        auto blockingQueue = device.makeQueue(queueKind::blocking);

        int const iterations = 20;
        std::vector<decltype(device.makeEvent())> events;
        events.reserve(iterations);

        for(int i = 0; i < iterations; ++i)
        {
            auto event = device.makeEvent();
            blockingQueue.enqueue(event);
            events.push_back(event);
        }

        // All events should be complete (blocking queue)
        for(auto const& ev : events)
        {
            CHECK(ev.isComplete());
        }

        // Create a new queue and verify it still works
        auto finalQueue = device.makeQueue(queueKind::blocking);
        auto finalEvent = device.makeEvent();
        finalQueue.enqueue(finalEvent);
        CHECK(finalEvent.isComplete());
    }

    SECTION("Blocking and non-blocking queue behavior")
    {
        auto blockingQueue = device.makeQueue(queueKind::blocking);
        auto nonBlockingQueue = device.makeQueue(queueKind::nonBlocking);

        int const iterations = 10;

        // Test blocking queue
        for(int i = 0; i < iterations; ++i)
        {
            auto event = device.makeEvent();
            blockingQueue.enqueue(event);
            CHECK(event.isComplete());
        }

        // Test non-blocking queue
        std::vector<decltype(device.makeEvent())> nonBlockingEvents;
        for(int i = 0; i < iterations; ++i)
        {
            auto event = device.makeEvent();
            nonBlockingQueue.enqueue(event);
            nonBlockingEvents.push_back(event);
        }

        onHost::wait(nonBlockingQueue);

        for(auto const& ev : nonBlockingEvents)
        {
            CHECK(ev.isComplete());
        }
    }
}

// Test for race condition in blocking queue event enqueue
TEMPLATE_LIST_TEST_CASE("blocking queue event race condition", "[bq][event][race]", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    auto blockingQueue = device.makeQueue(queueKind::blocking);

    // Test concurrent event operations to detect race conditions
    constexpr int numThreads = 4;
    constexpr int eventsPerThread = 10;

    std::vector<std::thread> threads;
    std::vector<std::vector<decltype(device.makeEvent())>> threadEvents(numThreads);

    // C++20 barrier: numThreads workers + 1 main thread = 5 total participants
    std::barrier syncPoint(numThreads + 1);

    // Create threads that will enqueue events concurrently
    for(int t = 0; t < numThreads; ++t)
    {
        // emplace thread with lambda that creates and enqueues events. Captures threadId by value
        threads.emplace_back(
            [&, threadId = t]()
            {
                // Worker thread waits at barrier for main thread to join
                syncPoint.arrive_and_wait();

                // Each thread creates and enqueues events
                for(int i = 0; i < eventsPerThread; ++i)
                {
                    auto event = device.makeEvent();

                    // Ensure event is set before waiting
                    blockingQueue.enqueue(event);

                    // Event should be immediately complete for blocking queue
                    CHECK(event.isComplete());

                    threadEvents[threadId].push_back(event);
                }
            });
    }

    // Main thread participates in barrier - releases all threads simultaneously
    syncPoint.arrive_and_wait();

    // Wait for all threads to complete
    for(auto& thread : threads)
    {
        thread.join();
    }

    // Verify all events are complete and consistent
    for(int t = 0; t < numThreads; ++t)
    {
        for(auto const& event : threadEvents[t])
        {
            CHECK(event.isComplete());
        }
        // Verify thread completed all expected operations (no crashes/hangs)
        CHECK(threadEvents[t].size() == eventsPerThread);
    }
}
