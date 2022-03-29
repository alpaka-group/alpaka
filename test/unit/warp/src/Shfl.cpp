/* Copyright 2022 David M. Rogers, Jan Stephan
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

#include <catch2/catch.hpp>

#include <cstdint>
#include <limits>

struct ShflSingleThreadWarpTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        if constexpr(alpaka::Dim<TAcc>::value > 0)
        {
            ALPAKA_CHECK(*success, alpaka::warp::getSize(acc) == 1);
            ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 42, -1) == 42);
        }
        else
        {
            ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 42, 0, 1) == 42);
        }
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 12, 0) == 12);
        float ans = alpaka::warp::shfl(acc, 3.3f, 0);
        ALPAKA_CHECK(*success, alpaka::math::floatEqualExactNoWarning(ans, 3.3f));
    }
};

struct ShflMultipleThreadWarpTestKernel
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

        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 42, 0) == 42);
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, threadIdxInWarp, 0) == 0);
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, threadIdxInWarp, 1) == 1);
        // Note the CUDA and HIP API-s differ on lane wrapping, but both agree it should not segfault
        // https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/14
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 5, -1) == 5);

        auto const epsilon = std::numeric_limits<float>::epsilon();

        // Test various widths
        for(int width = 1; width < warpExtent; width *= 2)
        {
            for(int idx = 0; idx < width; idx++)
            {
                int const off = width * (threadIdxInWarp / width);
                ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, threadIdxInWarp, idx, width) == idx + off);
                float const ans = alpaka::warp::shfl(acc, 4.0f - float(threadIdxInWarp), idx, width);
                float const expect = 4.0f - float(idx + off);
                ALPAKA_CHECK(*success, alpaka::math::abs(acc, ans - expect) < epsilon);
            }
        }

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(threadIdxInWarp >= warpExtent / 2)
            return;

        for(int idx = 0; idx < warpExtent / 2; idx++)
        {
            ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, threadIdxInWarp, idx) == idx);
            float const ans = alpaka::warp::shfl(acc, 4.0f - float(threadIdxInWarp), idx);
            float const expect = 4.0f - float(idx);
            ALPAKA_CHECK(*success, alpaka::math::abs(acc, ans - expect) < epsilon);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("shfl", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto const warpExtents = alpaka::getWarpSizes(dev);
    for(auto const warpExtent : warpExtents)
    {
        const auto scalar = Dim::value == 0 || warpExtent == 1;
        if(scalar)
        {
            alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(4));
            REQUIRE(fixture(ShflSingleThreadWarpTestKernel{}));
        }
        else
        {
            // Work around gcc 7.5 trying and failing to offload for OpenMP 4.0
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 5, 0)) && defined ALPAKA_ACC_ANY_BT_OMP5_ENABLED
            return;
#else
            using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
            auto const gridBlockExtent = alpaka::Vec<Dim, Idx>::all(2);
            // Enforce one warp per thread block
            auto blockThreadExtent = alpaka::Vec<Dim, Idx>::ones();
            blockThreadExtent[0] = static_cast<Idx>(warpExtent);
            auto const threadElementExtent = alpaka::Vec<Dim, Idx>::ones();
            auto workDiv = typename ExecutionFixture::WorkDiv{gridBlockExtent, blockThreadExtent, threadElementExtent};
            auto fixture = ExecutionFixture{workDiv};
            ShflMultipleThreadWarpTestKernel kernel;
            REQUIRE(fixture(kernel));
#endif
        }
    }
}
