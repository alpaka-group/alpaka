/* Copyright 2020 Sergei Bastrakov
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

#include <climits>
#include <cstdint>

class ActivemaskSingleThreadWarpTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        std::int32_t const warp_extent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warp_extent == 1);

        ALPAKA_CHECK(*success, alpaka::warp::activemask(acc) == 1u);
    }
};

class ActivemaskMultipleThreadWarpTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, std::uint64_t inactive_thread_idx) const -> void
    {
        std::int32_t const warp_extent = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, warp_extent > 1);

        // Test relies on having a single warp per thread block
        auto const block_extent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        ALPAKA_CHECK(*success, static_cast<std::int32_t>(block_extent.prod()) == warp_extent);
        auto const local_thread_idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const thread_idx_in_warp = static_cast<std::uint64_t>(alpaka::mapIdx<1u>(local_thread_idx, block_extent)[0]);

        if(thread_idx_in_warp == inactive_thread_idx)
            return;

        auto const actual = alpaka::warp::activemask(acc);
        using Result = decltype(actual);
        Result const all_active = static_cast<size_t>(warp_extent) == sizeof(Result) * CHAR_BIT
            ? ~Result{0u}
            : (Result{1} << warp_extent) - 1u;
        Result const expected = all_active & ~(Result{1} << inactive_thread_idx);
        ALPAKA_CHECK(*success, actual == expected);
    }
};

TEMPLATE_LIST_TEST_CASE("activemask", "[warp]", alpaka::test::TestAccs)
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
        ActivemaskSingleThreadWarpTestKernel kernel;
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
        ActivemaskMultipleThreadWarpTestKernel kernel;
        for(auto inactive_thread_idx = 0u; inactive_thread_idx < warp_extent; inactive_thread_idx++)
            REQUIRE(fixture(kernel, inactive_thread_idx));
#endif
    }
}
