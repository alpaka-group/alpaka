/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp "ballot" operation which collects votes from all lanes into a bitmask.
 * The "ballot" warp operation returns a bitmask where each bit represents whether that lane's predicate was true.
 * It's a collective operation that gathers boolean values from all participating threads into a single integer.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct BallotMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            // use runtime warp size to avoid compiler warning later
            uint32_t warpExtent = onAcc::warp::getSize<ALPAKA_TYPEOF(acc)>();
            /* We can not use a static_assert for testing because the compiler will evaluate the warp size during the
             * host parsing to what will result in false negatives */
            warpCheck(success, warpExtent >= 1u);

            auto const threadsPerBlock = static_cast<std::uint32_t>(acc[alpaka::layer::thread].count().product());
            // number of threads should be a multiple of the warp size
            warpCheck(success, threadsPerBlock % warpExtent == 0);

            using ResultType = decltype(onAcc::warp::ballot(acc, 42));
            // Limit the comparison mask to the bits the backend can physically store.
            auto const maxBits = static_cast<std::uint32_t>(sizeof(ResultType) * CHAR_BIT);
            auto const totalBits = std::min(warpExtent, maxBits);
            auto const allActive
                = totalBits == maxBits ? ~ResultType{0u} : (ResultType{1} << totalBits) - ResultType{1};

            // When every lane votes true, the ballot should mark all active bits.
            warpCheck(success, onAcc::warp::ballot(acc, 42) == allActive);
            // When every lane votes false, the ballot mask should be empty.
            warpCheck(success, onAcc::warp::ballot(acc, 0) == 0u);

            auto const lane = static_cast<std::uint32_t>(onAcc::warp::getLaneIdx(acc));
            if(lane >= warpExtent / 2u)
            {
                // Upper half of the warp stays inactive to validate masked ballots.
                return;
            }

            auto const activeLaneCount = static_cast<std::uint32_t>(warpExtent / 2u);
            for(std::uint32_t idx = 0u; idx < activeLaneCount; ++idx)
            {
                // Each active lane should toggle exactly its bit when voting true.
                auto const bitMask = static_cast<ResultType>(ResultType{1} << idx);
                // Exactly one lane voting true elevates its corresponding bit.
                // Example: active lanes {0,1,2,3}; choosing idx=2 makes lane 2 vote true while others vote false,
                // yielding 0b0100.
                warpCheck(success, onAcc::warp::ballot(acc, lane == idx ? 1 : 0) == bitMask);

                auto const expected = ((ResultType{1} << activeLaneCount) - ResultType{1}) & ~bitMask;
                // Everyone except the toggled lane voting true should set all other bits.
                // Example: active lanes {0,1,2,3}; choosing idx=2 makes predicates {1,1,0,1}, yielding 0b1011.
                warpCheck(success, onAcc::warp::ballot(acc, lane == idx ? 0 : 1) == expected);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp ballot captures predicate lanes", "[warp][ballot]", WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto deviceProperties = device.getDeviceProperties();
    auto const warpExtent = deviceProperties.warpSize;

    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);

    auto const blocks = Vec<std::uint32_t, 1u>{5u};
    auto const threads = Vec<std::uint32_t, 1u>{4u * warpExtent};

    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{BallotMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
