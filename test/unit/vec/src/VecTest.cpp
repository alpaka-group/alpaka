/**
 * \file
 * Copyright 2016 Erik Zenker
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

#include <alpaka/alpaka.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(vec)

//#############################################################################
//!
//#############################################################################
template<
    typename TDim,
    typename TSize>
struct NonAlpakaVec
{
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    operator ::alpaka::Vec<
        TDim,
        TSize>() const
    {
        using AlpakaVector = ::alpaka::Vec<
            TDim,
            TSize
        >;
        AlpakaVector result(AlpakaVector::zeros());

        for(TSize d(0); d < TDim::value; ++d)
        {
            result[TDim::value - 1 - d] = (*this)[d];
        }    

        return result;
    }
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    auto operator [](TSize /*idx*/) const
    -> TSize
    {
        return static_cast<TSize>(0);
    }
};

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    vec1DConstructionFromNonAlpakaVec)
{
    using Dim = alpaka::dim::DimInt<1u>;
    using Size = std::size_t;

    NonAlpakaVec<Dim, Size> nonAlpakaVec;
    static_cast<alpaka::Vec<Dim, Size> >(nonAlpakaVec);
}

BOOST_AUTO_TEST_SUITE_END()
