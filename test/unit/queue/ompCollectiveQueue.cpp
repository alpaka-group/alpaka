/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** This test is testing the special queue OmpCollectiveQueue.
 *
 * The queue is **NOT** part of official alpaka, therefore it must be included separately.
 * The test is skipped if alpaka tests are build without OpenMP support.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/api/host/OmpCollectiveQueue.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <vector>


using namespace alpaka;

// We test only the executor cpuOmpBlocks because we use the special queue OmpCollectiveQueue
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, std::make_tuple(exec::cpuOmpBlocks)))>;

#if ALPAKA_OMP
/** The number of threads used for the OpenMP parallel section */
constexpr uint32_t numOmpThreads = 3;

// used to validate the ompCollective queue indide of a parallel section
struct KernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto counter) const
    {
#    if ALPAKA_COMP_NVCC && ALPAKA_ARCH_PTX
        uint32_t ompThreadIdx = 0;
#    else
        uint32_t ompThreadIdx = omp_get_thread_num();
#    endif
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = ompThreadIdx;
            onAcc::atomicAdd(acc, &counter[i], 1u);
        }
    }
};

// used to validate the ompCollective queue outside of a parallel section
struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto counter) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = i;
            onAcc::atomicAdd(acc, &counter[i], 1u);
        }
    }
};

// test the queue ompCollective within a parallel section
void testCombinationOmpParallel(auto queue, auto exec, concepts::Vector auto extents)
{
    auto hBuff = onHost::allocHost<uint32_t>(extents);

    // take care invokeSingle can be called outside of a OpenMP parallel section
    auto hBuffCounter = onHost::omp::invokeSingle([&] { return onHost::allocHostLike(hBuff); });

    /* Test as many as possible queue methods within the parallel section, because all of them have a special
     * implementation.
     */
#    pragma omp parallel num_threads(numOmpThreads)
    {
        /* Any non queue function can be call as lambda via the invoke helper method.
         * After the kernel call the buffer contains the index of the thread which accessed the corresponding element.
         */
        auto dBuff = onHost::omp::invokeSingle([&] { return onHost::alloc<uint32_t>(queue.getDevice(), extents); });

        auto dBuffCounter = onHost::allocDeferred<uint32_t>(queue, extents);

        onHost::fill(queue, dBuff, 0u);
        onHost::memset(queue, dBuffCounter, 0u);

        // wait is required because memset and fill are non-blocking and concurrent
        onHost::wait(queue);
        auto numFrames = ALPAKA_TYPEOF(extents)::fill(1);
        /* We need as frames blocks as we have numOmpThreads threads to guarantee each thread is executing at least one
         * element. This is required for our validation.
         */
        numFrames.x() = numOmpThreads;
        queue.enqueue(
            onHost::FrameSpec{numFrames, ALPAKA_TYPEOF(extents)::fill(1), exec},
            KernelBundle{KernelND{}, dBuff, dBuffCounter});

        // wait is required because kernel enqueue is non-blocking and concurrent
        onHost::wait(queue);

        onHost::memcpy(queue, hBuff, dBuff);
        onHost::memcpy(queue, hBuffCounter, dBuffCounter);
    }
    // we need to wait because memcpy within the parallel region will not block
    onHost::wait(queue);

    /* Use REQUIRE instead of CHECK to avoid spamming the output if the results are wrong.
     * we create a histogram to check that all threads are used.
     */
    std::vector<uint32_t> threadCounter(numOmpThreads, 0u);
    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            threadCounter[hBuff[idx]]++;
            // fail if more or less than one invocation on the same data item is detected
            REQUIRE(1u == hBuffCounter[idx]);
        });
    // Each thread should be used by the kernel at least once
    for(auto const& count : threadCounter)
        REQUIRE(count != 0u);
}

// test the queue ompCollective outside a parallel section
void testCombinationNoParallel(auto queue, auto exec, concepts::Vector auto extents)
{
    auto dBuff = onHost::alloc<ALPAKA_TYPEOF(extents)>(queue.getDevice(), extents);
    auto dBuffCounter = onHost::alloc<uint32_t>(queue.getDevice(), extents);

    auto hBuff = onHost::allocHostLike(dBuff);
    auto hBuffCounter = onHost::allocHostLike(dBuffCounter);

    onHost::memset(queue, dBuff, 0u);
    onHost::fill(queue, dBuffCounter, 0u);

    /* This wait is not required because the queue ompCollective is blocking but for testing that wait is callable this
     * is important.
     */
    onHost::wait(queue);

    queue.enqueue(
        onHost::FrameSpec{extents, ALPAKA_TYPEOF(extents)::fill(1), exec},
        KernelBundle{IotaKernelND{}, dBuff, dBuffCounter});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::memcpy(queue, hBuffCounter, dBuffCounter);

    // Use REQUIRE instead of CHECK to avoid spamming the output if the results are wrong.
    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            REQUIRE(idx == hBuff[idx]);
            // fail if more or less than one invocation on the same data item is detected
            REQUIRE(1u == hBuffCounter[idx]);
        });
}

template<typename Queue, typename Exec, typename Extent>
void runTest(Queue& queue, Exec exec, Extent const& bufferExtents)
{
    DYNAMIC_SECTION(
        "OpenMP within parallel section: exec=" << onHost::demangledName(exec) << ", extent=" << bufferExtents)
    {
        // we need at least numOmpThreads elements to guarantee that each thread is executing at least one element.
        CHECK(bufferExtents.product() >= numOmpThreads);
        testCombinationOmpParallel(queue, exec, bufferExtents);
    }
    DYNAMIC_SECTION("No OpenMP section: exec=" << onHost::demangledName(exec) << ", extent=" << bufferExtents)
    {
        // we need at least numOmpThreads elements to guarantee that each thread is executing at least one element.
        CHECK(bufferExtents.product() >= numOmpThreads);
        testCombinationNoParallel(queue, exec, bufferExtents);
    }
}

/** The test is testing the queue ompCollective once within a parallel OpenMP section and once outside.
 *  In both cases it must behave like a blocking queue.
 */
TEMPLATE_LIST_TEST_CASE("OmpCollectiveQueue", "[queue][OmpCollectiveQueue]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue(::alpaka::queueKind::ompCollective);

    auto extents = std::make_tuple(
        // 1D
        Vec{numOmpThreads},
        Vec{997u},
        // 2D
        Vec{2u, 3u},
        Vec{101u, 103u},
        // 3D
        Vec{2u, 3u, 5u},
        Vec{43u, 47u, 53u},
        // 4D
        Vec{2u, 3u, 5u, 7u},
        Vec{47u, 43u, 17u, 13u});

    std::apply([&](auto const&... extent) { (runTest(queue, exec, extent), ...); }, extents);
}
#else
TEST_CASE("OmpCollectiveQueue", "[queue][OmpCollectiveQueue]")
{
    WARN("no OpenMP available: Disable OmpCollectiveQueue test");
    SUCCEED("SKIP OmpCollectiveQueue test");
}
#endif
