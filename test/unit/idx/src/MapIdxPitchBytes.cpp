/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/dev/Traits.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/dim/TestDims.hpp>

#include <catch2/catch.hpp>

TEMPLATE_LIST_TEST_CASE("mapIdxPitchBytes", "[idx]", alpaka::test::TestDims)
{
    using Dim = TestType;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const extent_nd
        = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();

    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Elem = std::uint8_t;
    auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);
    auto parent_view = alpaka::createView(dev_acc, static_cast<Elem*>(nullptr), extent_nd);

    auto const offset = Vec::all(4u);
    auto const extent = Vec::all(4u);
    auto const idx_nd = Vec::all(2u);
    auto view = alpaka::createSubView(parent_view, extent, offset);
    auto pitch = alpaka::getPitchBytesVec(view);

    auto const idx1d = alpaka::mapIdxPitchBytes<1u>(idx_nd, pitch);
    auto const idx1d_delta = alpaka::mapIdx<1u>(idx_nd + offset, extent_nd) - alpaka::mapIdx<1u>(offset, extent_nd);

    auto const idx_nd_result = alpaka::mapIdxPitchBytes<Dim::value>(idx1d, pitch);

    // linear index in pitched offset box should be the difference between
    // linear index in parent box and linear index of offset
    REQUIRE(idx1d == idx1d_delta);
    // roundtrip
    REQUIRE(idx_nd == idx_nd_result);
}
