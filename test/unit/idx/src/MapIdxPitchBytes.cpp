/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <alpaka/test/dim/TestDims.hpp>
#include <alpaka/test/Extent.hpp>

#include <catch2/catch.hpp>

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "mapIdxPitchBytes", "[idx]", alpaka::test::dim::TestDims)
{
    using Dim = TestType;
    using Idx = std::size_t;
    using Vec = alpaka::vec::Vec<Dim, Idx>;

    auto const extentNd(alpaka::vec::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());

    using Acc = alpaka::example::ExampleDefaultAcc<Dim, Idx>;
    using Dev = alpaka::dev::Dev<Acc>;
    using Elem = std::uint8_t;
    auto const devAcc = alpaka::pltf::getDevByIdx<Acc>(0u);
    alpaka::mem::view::ViewPlainPtr<Dev, Elem, Dim, Idx> parentView( nullptr, devAcc, extentNd );

    auto const offset(Vec::all(4u));
    auto const extent(Vec::all(4u));
    auto const idxNd(Vec::all(2u));
    alpaka::mem::view::ViewSubView<Dev, Elem, Dim, Idx> view( parentView, extent, offset );
    auto pitch = alpaka::mem::view::getPitchBytesVec(view);

    auto const idx1d(alpaka::idx::mapIdxPitchBytes<1u>(idxNd, pitch));
    auto const idx1dDelta(alpaka::idx::mapIdx<1u>(idxNd+offset, extentNd)
            - alpaka::idx::mapIdx<1u>(offset, extentNd));

    auto const idxNdResult(alpaka::idx::mapIdxPitchBytes<Dim::value>(idx1d, pitch));

    // linear index in pitched offset box should be the difference between
    // linear index in parent box and linear index of offset
    REQUIRE(idx1d == idx1dDelta);
    // roundtrip
    REQUIRE(idxNd == idxNdResult);
}
