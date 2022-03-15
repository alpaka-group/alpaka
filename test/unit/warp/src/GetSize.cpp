/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
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
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, std::int32_t expectedWarpSize) const -> void
    {
        std::int32_t const actualWarpSize = alpaka::warp::getSize(acc);
        ALPAKA_CHECK(*success, actualWarpSize == expectedWarpSize);
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
    auto const warpSizes = alpaka::getWarpSizes(dev);
    Idx const gridThreadExtentPerDim = 8;
    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::all(gridThreadExtentPerDim));
    GetSizeTestKernel kernel;
    auto success = false;
    for(auto const expectedWarpSize : warpSizes)
    {
        // ensure that at least one of the supported warp sizes is used in the kernel
        success = success || fixture(kernel, static_cast<std::int32_t>(expectedWarpSize));
        if(success)
            break; // don't do more work than necessary
    }
    REQUIRE(success);
}
