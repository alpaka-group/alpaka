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

#include <cstdint>

class GetSizeTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, std::int32_t expected_warp_size) const -> void
    {
        std::int32_t const actual_warp_size = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, actual_warp_size == expected_warp_size);
    }
};

TEMPLATE_LIST_TEST_CASE("getSize", "[warp]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto const expected_warp_size = static_cast<int>(alpaka::getWarpSize(dev));
    Idx const grid_thread_extent_per_dim = 8;
    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(grid_thread_extent_per_dim));
    GetSizeTestKernel kernel;
    REQUIRE(fixture(kernel, expected_warp_size));
}
