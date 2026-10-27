/* Copyright 2025 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci, Aurora Perego, Mehmet
 * Yusufoglu, René Widera SPDX-License-Identifier: MPL-2.0
 */

/** @file Tests the warp "shfl_up" (shuffle up) operation which shifts values toward lower-numbered lanes.
 * The "shfl_up" warp operation allows each thread to read a value from a lane at a fixed offset above.
 * It's a data exchange operation useful for suffix scans and reverse reduction patterns within a warp.
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
    struct ShflUpMultiThreadKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            auto const warpExtent = static_cast<std::int32_t>(onAcc::warp::getSize(acc));
            warpCheck(success, warpExtent >= 1);

            auto const threadsPerBlock = static_cast<std::int32_t>(acc[alpaka::layer::thread].count().product());
            warpCheck(success, threadsPerBlock % warpExtent == 0);

            auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));

            // With zero offset, every lane should see the source literal unchanged.
            warpCheck(success, onAcc::warp::shflUp(acc, 42, 0u) == 42);
            // A zero-width shuffle leaves each lane's original value intact.
            // For example, lane 2 with offset 0 should see lane 2's own value.
            warpCheck(success, onAcc::warp::shflUp(acc, lane, 0u) == lane);
            // Offset of one shifts values toward lower indices, clamping at the partition boundary.
            // For example, lane 0 with offset 1 should see lane 0's own value, lane 1 should see lane 0's value, and
            // so on.
            warpCheck(success, onAcc::warp::shflUp(acc, lane, 1u) == (lane - 1 >= 0 ? lane - 1 : lane));

            auto const epsilon = std::numeric_limits<float>::epsilon();
            for(int width = 1; width < warpExtent; width *= 2)
            {
                // Exercise each logical partition width independently.
                for(int idx = 0; idx < width; ++idx)
                {
                    auto const sectionStart = width * (lane / width);
                    auto const shuffled = onAcc::warp::shflUp(
                        acc,
                        lane,
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expectedInt = (lane - idx >= sectionStart) ? lane - idx : lane;
                    // Each lane should see the value from the lane at the offset above, or clamp to its own value if
                    // out of range. For example, if lane 2 with offset 1 should see lane 1's value, but if lane 1 is
                    // the first lane in the partition, it sees its own value
                    warpCheck(success, shuffled == expectedInt);

                    auto const ans = onAcc::warp::shflUp(
                        acc,
                        4.0f - static_cast<float>(lane),
                        static_cast<std::uint32_t>(idx),
                        static_cast<std::uint32_t>(width));
                    auto const expect = (lane - idx >= sectionStart) ? 4.0f - static_cast<float>(lane - idx)
                                                                     : 4.0f - static_cast<float>(lane);
                    warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
                }
            }

            if(lane >= warpExtent / 2)
            {
                // Suppress half the warp to emulate a masked collective.
                return;
            }

            for(int idx = 0; idx < warpExtent / 2; ++idx)
            {
                // Remaining active lanes should shift values toward lower indices.
                auto const shuffled = onAcc::warp::shflUp(acc, lane, static_cast<std::uint32_t>(idx));
                auto const ans
                    = onAcc::warp::shflUp(acc, 4.0f - static_cast<float>(lane), static_cast<std::uint32_t>(idx));
                auto const expect
                    = (lane - idx >= 0) ? 4.0f - static_cast<float>(lane - idx) : 4.0f - static_cast<float>(lane);
                // Each lane should see the value from the lane at the offset above within the active sub-group.
                warpCheck(success, shuffled == (lane - idx >= 0 ? lane - idx : lane));
                warpCheck(success, alpaka::math::abs(ans - expect) < epsilon);
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE("warp shflUp shifts toward lower lanes", "[warp][shfl_up]", WarpTestBackends)
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
    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{ShflUpMultiThreadKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
