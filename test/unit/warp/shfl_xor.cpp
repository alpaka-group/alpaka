/* Copyright 2025 Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp "shfl_xor" (shuffle XOR) operation which exchanges values according to XOR-based lane pairing.
 * The "shfl_xor" warp operation allows each thread to read from a lane whose ID is the XOR of its own ID with a mask.
 * It's a data exchange operation commonly used in butterfly reduction and FFT-like parallel algorithms.
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    struct ShflXorMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::int32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent >= 1);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock % warpExtent == 0);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));
            // Exercise trivial zero-offset and max-offset cases.
            // For zero offset, each lane should see its own value.
            warpCheck(success, onAcc::warp::shflXor(acc, 42, 0u) == 42);
            // For zero offset, each lane should see its own value.
            warpCheck(success, onAcc::warp::shflXor(acc, lane, 0u) == lane);

            // For offset one, each lane should xor with 1 to find its partner.
            // For example, lane 0 with offset 1 should see lane 1's value, lane 1 should see lane 0's value, and so
            // on.
            auto shuffleOneMaskResult = onAcc::warp::shflXor(acc, lane, 1u);
            warpCheck(
                success,
                shuffleOneMaskResult == (lane ^ 1) || (warpExtent == 1 && shuffleOneMaskResult == lane));

            // Max offset should behave like zero offset since no lanes exist beyond the warp size.
            // For example, lane 2 with max offset should see lane 2's own value.
            warpCheck(success, onAcc::warp::shflXor(acc, 5, std::numeric_limits<std::uint32_t>::max()) == 5);

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(int width = 1; width < warpExtent; width *= 2)
            {
                // Test every xor distance inside the advertised subgroup width.
                for(int idx = 0; idx < width; ++idx)
                {
                    auto const shuffled = onAcc::warp::shflXor(
                        acc,
                        lane,
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    warpCheck(success, shuffled == (lane ^ idx));

                    auto const ans = onAcc::warp::shflXor(
                        acc,
                        4.0f - static_cast<float>(lane),
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expect = 4.0f - static_cast<float>(lane ^ idx);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(lane >= warpExtent / 2)
            {
                // Deactivate the upper half to probe masked partners.
                return;
            }

            for(int idx = 0; idx < warpExtent / 2; ++idx)
            {
                // Remaining lanes must xor-pair with the expected partner.
                warpCheck(success, onAcc::warp::shflXor(acc, lane, static_cast<std::uint32_t>(idx)) == (lane ^ idx));
                auto const ans
                    = onAcc::warp::shflXor(acc, 4.0f - static_cast<float>(lane), static_cast<std::uint32_t>(idx));
                auto const expect = 4.0f - static_cast<float>(lane ^ idx);
                warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shflXor exchanges partner lanes", "[warp][shfl_xor]", WarpTestBackends)
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
    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{ShflXorMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
