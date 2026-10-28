/* Copyright 2022 Jan Stephan, Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */
/** @file
 * Unit tests for non-blocking memory visibility fences (memFence).
 * Contains:
 *  - ProducerConsumerKernel: publication pattern using device-scope fences.
 *  - BlockSharedMemOrderKernel: shared-memory ordering using block-scope fences.
 */

#include "alpaka/onAcc/memoryOrder.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

// Producer-Consumer kernel documentation:
//  - Producer (thread 0) publishes a payload (value = iteration index) to global memory,
//    then issues a device-scope memFence to ensure the data write becomes visible
//    before setting the corresponding ready flag to 1.
//  - Consumer (thread 1) busy-waits on the ready flag becoming 1, then issues its own
//    device-scope fence before reading the payload. This is like a typical acquire
//    after producer's release.
//
// If the fence were missing on producer side, a reordering (or visibility delay) could
// allow the consumer to observe ready==1 but still see an old payload value. This would
// not be caught by a simple correctness test, so we count mismatches. Reordering is not
// guaranteed to manifest every run on all hardware; this encodes the canonical
// correctness pattern and will fail if a backend ever weakens ordering.
struct ProducerConsumerKernel
{
    template<class T_Acc, class T_Val, class T_Flag, class T_Mis>
    ALPAKA_FN_ACC void operator()(
        T_Acc const& acc,
        T_Val payload,
        T_Flag readyFlags,
        T_Mis mismatches,
        uint32_t const iterations) const
    {
        using namespace alpaka::onAcc;
        auto [tid] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        // Only two active participants: 0 = producer, 1 = consumer.
        if(!(tid == 0 || tid == 2))
            // other threads exit fast.
            return;

        for(uint32_t i = 0; i < iterations; ++i)
        {
            // only for tid == 0, first thread is producer
            if(tid == 0u)
            {
                // Publish payload value first.
                atomicExch(acc, &payload[i], i);
                // Ensure payload write is visible before flag store.
                memFence(acc, scope::device, order::release);
                atomicExch(acc, &readyFlags[i], 1u);
            }
            // consumer, only for tid == 1, second thread is consumer
            else
            {
                // Spin until flag is set at each iteration.
                while(atomicCas(acc, &readyFlags[i], 0u, 0u) == 0u)
                { /* busy wait */
                }
                // Acquire fence: prevents compiler/hardware from speculatively reading
                // payload before observing the flag. This is the "Acquire" part; which is completing the
                // release-acquire pair The busy-wait loop guarantees thread 1 doesn't leave the loop until it
                // has observed flag==1, but it does not by itself force any refresh/invalidation of the cache
                // line holding payload array values, nor prevent the compiler from reordering a payload-read
                // above the while-loop unless the flag read is treated as a dependency (atomic/volatile) and
                // an acquire fence follows.

                memFence(acc, scope::device, order::acquire);
                auto v = payload[i];
                if(v != i)
                {
                    // Increment mismatch counter (benign race: only consumer writes on mismatch).
                    onAcc::atomicAdd(acc, mismatches.data(), 1u);
                }
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("memFence producer-consumer publication", "[memFence][producer-consumer]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    // modest to keep test fast.
    constexpr uint32_t iterations = 10u;

    auto payload = onHost::alloc<uint32_t>(device, Vec{iterations});
    auto flags = onHost::alloc<uint32_t>(device, Vec{iterations});
    auto mis = onHost::alloc<uint32_t>(device, Vec{1u});

    auto hFlags = onHost::allocHostLike(flags);
    auto hMis = onHost::allocHostLike(mis);

    // Init device buffers.
    meta::ndLoopIncIdx(Vec{iterations}, [&](auto i) { hFlags[i] = 0u; });
    hMis[Vec{0u}] = 0u;
    // Copy initial state to device.
    onHost::memcpy(queue, flags, hFlags);
    onHost::memcpy(queue, mis, hMis);

    /* Launch with exactly three logical threads using a FrameSpec (extent = 3, frameSize = 1).
     * We would for the test only require 2 threads but for unknown reason on OneAPi with an Intel GPU (tested with ARC
     * A770 + icpx 2025.2) the code deadlock. The reason for it is that it looks like that OneApi is starting 2 native
     * sycl threads blocked groups of 1 thread within a single group and not two independent groups. Due to the reason
     * that if conditions are executed lock step first the branch with the busy wait is executed and then never the
     * branch for thread with id zero. Using 3 groups aka 3 thread blocks and working with thread 0 and 2 worked fine.
     *
     * @todo: find out if this strange behaviour is an bug (most likely) or somewhere documented in SYCL or OneApi.
     */
    queue.enqueue(
        onHost::ThreadSpec{3, 1, exec},
        KernelBundle{ProducerConsumerKernel{}, payload, flags, mis, iterations});
    onHost::wait(queue);

    // Copy mismatch counter back.
    onHost::memcpy(queue, hMis, mis);
    onHost::wait(queue);

    CHECK(hMis[Vec{0u}] == 0u);
}

struct DeviceFenceTestKernelWriter
{
    template<typename T_Acc>
    ALPAKA_FN_ACC auto operator()(T_Acc const& acc, int volatile* vars) const -> void
    {
        auto const [idx] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        // Use a single writer thread
        if(idx == 0)
        {
            vars[0] = 10;
            onAcc::memFence(acc, onAcc::scope::Device{}, onAcc::order::release);
            vars[1] = 20;
        }
    }
};

struct DeviceFenceTestKernelReader
{
    template<typename T_Acc>
    ALPAKA_FN_ACC auto operator()(T_Acc const& acc, auto successFlag, int volatile* vars) const -> void
    {
        auto const [idx] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        // Use a single reader thread
        if(idx == 0)
        {
            auto const b = vars[1];
            onAcc::memFence(acc, onAcc::scope::Device{}, onAcc::order::acquire);
            auto const a = vars[0];

            // If the fence is working correctly, the following case can never happen
            if(a == 1 && b == 20)
            {
                // mark failure atomically to handle concurrent writes
                onAcc::atomicExch(acc, &successFlag[0], 0u);
            }
        }
    }
};

struct DeviceFenceTestKernel
{
    template<typename T_Acc>
    ALPAKA_FN_ACC auto operator()(T_Acc const& acc, auto successFlag, int volatile* vars) const -> void
    {
        auto const [idx] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        // Global thread 0 is producer
        if(idx == 0)
        {
            vars[0] = 10;
            onAcc::memFence(acc, onAcc::scope::Device{}, onAcc::order::release);
            vars[1] = 20;
        }

        auto const b = vars[1];
        onAcc::memFence(acc, onAcc::scope::Device{}, onAcc::order::acquire);
        auto const a = vars[0];

        // If the fence is working correctly, the following case can never happen
        if(a == 1 && b == 20)
        {
            // mark failure atomically to handle concurrent writes
            onAcc::atomicExch(acc, &successFlag[0], 0u);
        }
    }
};

// -------------------------------------------------------------------------------------------------
// Block shared-memory ordering test:
// Validates that writes to dynamic shared memory from one thread become visible to sibling threads
// in the same block after a block-scope fence. Pattern mirrors legacy BlockFenceTestKernel from the
// original alpaka repo but adapted to memFence API and Catch2 style.
struct BlockSharedMemOrderKernel
{
    // number of bytes of dynamic shared memory required by this kernel
    static constexpr ::std::uint32_t dynSharedMemBytes = 2u * sizeof(int);

    ALPAKA_FN_ACC void operator()(auto const& acc, auto successFlag) const
    {
        using namespace alpaka::onAcc;
        // need space for 2 ints
        auto* shared = getDynSharedMem<int>(acc);

        for(auto [tid] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::threadsInGrid))
        {
            // Initialize once by the producer thread with id 0.
            if(tid == 0u)
            {
                // A
                shared[0] = 1;
                // B
                shared[1] = 2;
            }
            syncBlockThreads(acc);

            // Producer thread with id 0 updates A then fences then updates B.
            if(tid == 0u)
            {
                // publish new A
                shared[0] = 10;
                // ensure visibility of A before B write
                memFence(acc, scope::block, onAcc::order::release);
                // publish B
                shared[1] = 20;
            }

            // allow consumer threads to observe writes after fence ordering
            syncBlockThreads(acc);

            // All threads perform the read/validation (any non-producer could be consumer)
            auto b = shared[1];
            // acquire side
            memFence(acc, scope::block, onAcc::order::acquire);
            auto a = shared[0];

            // Forbidden outcome: observe updated B (20) but stale A (1)
            if(a == 1 && b == 20)
            {
                // mark failure atomically to handle concurrent writes
                onAcc::atomicExch(acc, &successFlag[0], 0u);
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("memFence block shared-memory ordering", "[memFence][block-shared]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    // success flag: 1 = pass, 0 = failure detected
    auto flag = onHost::allocUnified<uint32_t>(device, Vec{1u});

    int const numElements = 2;
    auto varsHost = onHost::allocMapped<int>(device, numElements);
    auto varsDev = onHost::alloc<int>(device, numElements);
    varsHost[0] = 1;
    varsHost[1] = 2;

    {
        flag[0u] = 1u;
        queue.enqueue(onHost::FrameSpec{1, 2, exec}, KernelBundle{BlockSharedMemOrderKernel{}, flag});
        onHost::wait(queue);
        CHECK(flag[0u] == 1u);
    }

    {
        // Run a single kernel, testing a memory fence in global memory across threads in different blocks
        onHost::memcpy(queue, varsDev, varsHost);
        onHost::wait(queue);
        flag[0u] = 1u;
        // Device-scope variant, use thread specification to guarantee that we have two thread blocks
        // A frame specification is allowed silently to change the number of real thread blocks and the block size
        queue.enqueue(onHost::ThreadSpec{2, 1, exec}, KernelBundle{DeviceFenceTestKernel{}, flag, varsDev.data()});
        onHost::wait(queue);
        CHECK(flag[0u] == 1u);
    }

    {
        // Run two kernels in parallel, in two different queues on the same device, testing a memory fence
        // in global memory across threads in different grids
        onHost::memcpy(queue, varsDev, varsHost);
        onHost::wait(queue);
        auto queue1 = device.makeQueue();

        flag[0u] = 1u;
        queue.enqueue(onHost::ThreadSpec{1, 1, exec}, KernelBundle{DeviceFenceTestKernelWriter{}, varsDev.data()});
        queue1.enqueue(
            onHost::ThreadSpec{1, 1, exec},
            KernelBundle{DeviceFenceTestKernelReader{}, flag, varsDev.data()});
        onHost::wait(queue);
        onHost::wait(queue1);
        CHECK(flag[0u] == 1u);
    }
}
