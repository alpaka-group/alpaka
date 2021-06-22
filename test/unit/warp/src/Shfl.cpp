/* Copyright 2021 David M. Rogers
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/warp/Traits.hpp>

#include <catch2/catch.hpp>

#include <cstdint>
#include <limits>

class ShflSingleThreadWarpTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        std::int32_t const warp_extent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warp_extent == 1);

        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 12, 0) == 12);
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 42, -1) == 42);
        float ans = alpaka::warp::shfl(acc, 3.3f, 0);
        ALPAKA_CHECK(*success, alpaka::math::abs(acc, ans - 3.3f) < 1e-8f);
    }
};

class ShflMultipleThreadWarpTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        auto const local_thread_idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const block_extent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        std::int32_t const warp_extent = alpaka::warp::getSize(acc);
        // Test relies on having a single warp per thread block
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(block_extent.prod()) == warp_extent);
        auto const thread_idx_in_warp = std::int32_t(alpaka::mapIdx<1u>(local_thread_idx, block_extent)[0]);

        ALPAKA_CHECK(*success, warp_extent > 1);

        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 42, 0) == 42);
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, thread_idx_in_warp, 0) == 0);
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, thread_idx_in_warp, 1) == 1);
        // Note the CUDA and HIP API-s differ on lane wrapping, but both agree it should not segfault
        // https://github.com/ROCm-Developer-Tools/HIP-CPU/issues/14
        ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, 5, -1) == 5);

        auto const epsilon = std::numeric_limits<float>::epsilon();

        // Test various widths
        for(int width = 1; width < warp_extent; width *= 2)
        {
            for(int idx = 0; idx < width; idx++)
            {
                int const off = width * (thread_idx_in_warp / width);
                ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, thread_idx_in_warp, idx, width) == idx + off);
                float const ans = alpaka::warp::shfl(acc, 4.0f - float(thread_idx_in_warp), idx, width);
                float const expect = 4.0f - float(idx + off);
                ALPAKA_CHECK(*success, alpaka::math::abs(acc, ans - expect) < epsilon);
            }
        }

        // Some threads quit the kernel to test that the warp operations
        // properly operate on the active threads only
        if(thread_idx_in_warp >= warp_extent / 2)
            return;

        for(int idx = 0; idx < warp_extent / 2; idx++)
        {
            ALPAKA_CHECK(*success, alpaka::warp::shfl(acc, thread_idx_in_warp, idx) == idx);
            float const ans = alpaka::warp::shfl(acc, 4.0f - float(thread_idx_in_warp), idx);
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
    auto const warp_extent = alpaka::getWarpSize(dev);
    if(warp_extent == 1)
    {
        Idx const grid_thread_extent_per_dim = 4;
        alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(grid_thread_extent_per_dim));
        ShflSingleThreadWarpTestKernel kernel;
        REQUIRE(fixture(kernel));
    }
    else
    {
        // Work around gcc 7.5 trying and failing to offload for OpenMP 4.0
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 5, 0)) && defined ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        return;
#else
        using ExecutionFixture = alpaka::test::KernelExecutionFixture<Acc>;
        auto const grid_block_extent = alpaka::Vec<Dim, Idx>::all(2);
        // Enforce one warp per thread block
        auto block_thread_extent = alpaka::Vec<Dim, Idx>::ones();
        block_thread_extent[0] = static_cast<Idx>(warp_extent);
        auto const thread_element_extent = alpaka::Vec<Dim, Idx>::ones();
        auto work_div = typename ExecutionFixture::WorkDiv{grid_block_extent, block_thread_extent, thread_element_extent};
        auto fixture = ExecutionFixture{work_div};
        ShflMultipleThreadWarpTestKernel kernel;
        REQUIRE(fixture(kernel));
#endif
    }
}
