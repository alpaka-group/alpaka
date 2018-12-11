/**
 * \file
 * Copyright 2017 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>


//#############################################################################
//! 1D: (17)
//! 2D: (17, 14)
//! 3D: (17, 14, 11)
//! 4D: (17, 14, 11, 8)
template<
    std::size_t Tidx>
struct CreateExtentVal
{
    //-----------------------------------------------------------------------------
    template<
        typename TIdx>
    ALPAKA_FN_HOST_ACC static auto create(
        TIdx)
    -> TIdx
    {
        return  static_cast<TIdx>(17u - (Tidx*3u));
    }
};

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TDim >
void operator()()
{
    using Idx = std::size_t;
    using Vec = alpaka::vec::Vec<TDim, Idx>;

    auto const extentNd(alpaka::vec::createVecFromIndexedFnWorkaround<TDim, Idx, CreateExtentVal>(Idx()));
    auto const idxNd(extentNd - Vec::all(4u));

    auto const idx1d(alpaka::idx::mapIdx<1u>(idxNd, extentNd));

    auto const idxNdResult(alpaka::idx::mapIdx<TDim::value>(idx1d, extentNd));

    REQUIRE(idxNd == idxNdResult);
}
};

TEST_CASE( "mapIdx", "[idx]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestDims >( TestTemplate() );
}
