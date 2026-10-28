/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

struct AtomicIncrementKernel
{
    template<typename T_Acc, typename T_Counter>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, T_Counter counter, auto repetitions) const
    {
        /** To make the behavior consistent across all backends, we iterate over the full frame extent and use a for
         * loop and set range to totalFrameSpecExtent. If a single `alpaka::onAcc::atomicAdd(acc, &(counter[Vec{0u}]),
         * 1u);` is used instead of a for loop, cpuSerial backend returns 1 instead of numberOfBlocks. Makes only one
         * addition.
         */
        for(auto const& idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{repetitions}))
        {
            alpaka::unused(idx);
            alpaka::onAcc::atomicAdd(acc, &(counter[Vec{0u}]), 1u);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("cpu atomic add increments", "[executor][atomic]", TestApis)
{
    /** Launch each enabled host executor and ensure the atomic increment succeeds on the device counter.
     * This guards against regressions in the shared stlAtomic backend used by the TBB executor path.
     */
    using namespace alpaka;

    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue(queueKind::blocking);

    auto counterDev = onHost::alloc<std::uint32_t>(device, Vec{1u});
    auto counterHost = onHost::allocHostLike(counterDev);

    onHost::memset(queue, counterDev, 0);

    constexpr Vec blocks = Vec{64u};
    constexpr Vec threads = Vec{1u};

    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{AtomicIncrementKernel{}, counterDev, blocks});
    onHost::memcpy(queue, counterHost, counterDev);
    onHost::wait(queue);

    REQUIRE(counterHost[0] == blocks.x());
}
