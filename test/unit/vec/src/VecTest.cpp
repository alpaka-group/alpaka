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

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(
    basicVecTraits)
{
    using Dim = alpaka::dim::DimInt<3u>;
    using Size = std::size_t;
    using Vec = alpaka::Vec<Dim, Size>;

    auto const vec(
        Vec(
            static_cast<std::size_t>(0u),
            static_cast<std::size_t>(8u),
            static_cast<std::size_t>(15u)));

    //-----------------------------------------------------------------------------
    // alpaka::vec::subVecFromIndices
    {
        using IdxSequence =
            alpaka::meta::IntegerSequence<
                std::size_t,
                0u,
                Dim::value -1u,
                0u>;
        auto const vecSubIndices(
            alpaka::vec::subVecFromIndices<
                IdxSequence>(
                    vec));

        BOOST_REQUIRE_EQUAL(vecSubIndices[0u], vec[0u]);
        BOOST_REQUIRE_EQUAL(vecSubIndices[1u], vec[Dim::value -1u]);
        BOOST_REQUIRE_EQUAL(vecSubIndices[2u], vec[0u]);
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::subVecBegin
    {
        using DimSubVecEnd =
            alpaka::dim::DimInt<2u>;
        auto const vecSubBegin(
            alpaka::vec::subVecBegin<
                DimSubVecEnd>(
                    vec));

        for(typename Dim::value_type i(0); i < DimSubVecEnd::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecSubBegin[i], vec[i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::subVecEnd
    {
        using DimSubVecEnd =
            alpaka::dim::DimInt<2u>;
        auto const vecSubEnd(
            alpaka::vec::subVecEnd<
                DimSubVecEnd>(
                    vec));

        for(typename Dim::value_type i(0); i < DimSubVecEnd::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecSubEnd[i], vec[Dim::value - DimSubVecEnd::value + i]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::cast
    {
        using SizeCast = std::uint16_t;
        auto const vecCast(
            alpaka::vec::cast<
                SizeCast>(
                    vec));

        using VecCast = typename std::decay<decltype(vecCast)>::type;
        static_assert(
            std::is_same<
                alpaka::size::Size<VecCast>,
                SizeCast
            >::value,
            "The size type of the casted vec is wrong");

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecCast[i], static_cast<SizeCast>(vec[i]));
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::vec::reverse
    {
        auto const vecReverse(
            alpaka::vec::reverse(
                vec));

        for(typename Dim::value_type i(0); i < Dim::value; ++i)
        {
            BOOST_REQUIRE_EQUAL(vecReverse[i], vec[Dim::value - 1u - i]);
        }
    }
}

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
