/* Copyright 2023 Aurora Perego
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/math/FloatEqualExact.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/warp/Traits.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

struct ShflUpSingleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(alpaka::Dim<TAcc>::value > 0)
        {
            ALPAKA_CHECK(*success, alpaka::warp::getSize(acc) == 1);
            ALPAKA_CHECK(*success, alpaka::warp::shfl_up(acc, 42, 0) == 42);
        }
        else
        {
            ALPAKA_CHECK(*success, alpaka::warp::shfl_up(acc, 42, 0, 1) == 42);
        }
        ALPAKA_CHECK(*success, alpaka::warp::shfl_up(acc, 12, 0) == 12);
        float ans = alpaka::warp::shfl_up(acc, 3.3f, 0);
        ALPAKA_CHECK(*success, alpaka::math::floatEqualExactNoWarning(ans, 3.3f));
    }
};

template<std::uint32_t TWarpSize>
struct ShflUpMultipleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        std::int32_t const warpExtent = alpaka::warp::getSize(acc);
        // Test relies on having a single warp per thread block
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(blockExtent.prod()) == warpExtent);
        auto const threadIdxInWarp = std::int32_t(alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0]);

        ALPAKA_CHECK(*success, warpExtent > 1);

        ALPAKA_CHECK(*success, alpaka::warp::shfl_up(acc, 42, 0) == 42);
        ALPAKA_CHECK(*success, alpaka::warp::shfl_up(acc, threadIdxInWarp, 0) == threadIdxInWarp);
        ALPAKA_CHECK(
            *success,
            alpaka::warp::shfl_up(acc, threadIdxInWarp, 1)
                == (threadIdxInWarp - 1 >= 0 ? threadIdxInWarp - 1 : threadIdxInWarp));

        auto const epsilon = std::numeric_limits<float>::epsilon();

        // Test various widths
        for(int width = 1; width < warpExtent; width *= 2)
        {
            for(int idx = 0; idx < width; idx++)
            {
                int const off = width * (threadIdxInWarp / width);
                ALPAKA_CHECK(
                    *success,
                    alpaka::warp::shfl_up(acc, threadIdxInWarp, static_cast<std::uint32_t>(idx), width)
                        == ((threadIdxInWarp - idx >= off) ? threadIdxInWarp - idx : threadIdxInWarp));
                float const ans = alpaka::warp::shfl_up(
                    acc,
                    4.0f - float(threadIdxInWarp),
                    static_cast<std::uint32_t>(idx),
                    width);
                float const expect
                    = ((threadIdxInWarp - idx >= off) ? (4.0f - float(threadIdxInWarp - idx))
                                                      : (4.0f - float(threadIdxInWarp)));
                ALPAKA_CHECK(*success, alpaka::math::abs(acc, ans - expect) < epsilon);
            }
        }

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(threadIdxInWarp >= warpExtent / 2)
            return;

        for(int idx = 0; idx < warpExtent / 2; idx++)
        {
            ALPAKA_CHECK(
                *success,
                alpaka::warp::shfl_up(acc, threadIdxInWarp, static_cast<std::uint32_t>(idx))
                    == ((threadIdxInWarp - idx >= 0) ? (threadIdxInWarp - idx) : threadIdxInWarp));
            float const ans
                = alpaka::warp::shfl_up(acc, 4.0f - float(threadIdxInWarp), static_cast<std::uint32_t>(idx));
            float const expect
                = ((threadIdxInWarp - idx >= 0) ? (4.0f - float(threadIdxInWarp - idx))
                                                : (4.0f - float(threadIdxInWarp)));
            ALPAKA_CHECK(*success, alpaka::math::abs(acc, ans - expect) < epsilon);
        }
    }
};

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<ShflUpMultipleThreadWarpTestKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};

TEMPLATE_LIST_TEST_CASE("shfl_up", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platform = alpaka::Platform<Acc>{};
    Dev const dev(alpaka::getDevByIdx(platform, 0u));
    auto const warpExtents = alpaka::getWarpSizes(dev);
    for(auto const warpExtent : warpExtents)
    {
        auto const scalar = Dim::value == 0 || warpExtent == 1;
        if(scalar)
        {
            alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(4));
            REQUIRE(fixture(ShflUpSingleThreadWarpTestKernel{}));
        }
        else
        {
            using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
            auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
            // Enforce one warp per thread block
            auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
            blockThreadExtent[0] = static_cast<Idx>(warpExtent);
            auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
            auto workDiv = typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
            auto fixture = ExecutionFixture{workDiv};
            if(warpExtent == 4)
            {
                REQUIRE(fixture(ShflUpMultipleThreadWarpTestKernel<4>{}));
            }
            else if(warpExtent == 8)
            {
                REQUIRE(fixture(ShflUpMultipleThreadWarpTestKernel<8>{}));
            }
            else if(warpExtent == 16)
            {
                REQUIRE(fixture(ShflUpMultipleThreadWarpTestKernel<16>{}));
            }
            else if(warpExtent == 32)
            {
                REQUIRE(fixture(ShflUpMultipleThreadWarpTestKernel<32>{}));
            }
            else if(warpExtent == 64)
            {
                REQUIRE(fixture(ShflUpMultipleThreadWarpTestKernel<64>{}));
            }
        }
    }
}
