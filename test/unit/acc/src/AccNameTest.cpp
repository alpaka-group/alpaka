/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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
#include <alpaka/test/acc/Acc.hpp>      // EnabledAccs

#include <boost/test/unit_test.hpp>

#include <iostream>                     // std::cout
#include <tuple>                        // std::tuple
#include <type_traits>                  // std::is_same
#include <cstdint>

BOOST_AUTO_TEST_SUITE(acc)

//#############################################################################
//! A std::tuple holding dimensions.
//#############################################################################
using TestDimsTypeList =
    std::tuple<
        alpaka::dim::DimInt<1u>,
        alpaka::dim::DimInt<2u>,
        alpaka::dim::DimInt<3u>,
        alpaka::dim::DimInt<4u>>;

//#############################################################################
//! A std::tuple holding size types.
//#############################################################################
using TestSizesTypeList =
    std::tuple<
        std::size_t,
        std::int64_t,
        std::uint64_t,
        std::int32_t,
        std::uint32_t,
        std::int16_t,
        std::uint16_t,
        std::int8_t,
        std::uint8_t>;

//#############################################################################
//! A std::tuple holding multiple std::tuple consisting of a dimension and a size type.
//!
//! TestParamSets =
//!     tuple<
//!         tuple<Dim1,Size1>,
//!         tuple<Dim2,Size1>,
//!         tuple<Dim3,Size1>,
//!         ...,
//!         tuple<DimN,SizeN>>
//#############################################################################
using TestParamSets =
    alpaka::meta::CartesianProduct<
        std::tuple,
        TestDimsTypeList,
        TestSizesTypeList
    >;

//#############################################################################
//! Transforms a std::tuple holding a dimension and a size type to a fully specialized accelerator.
//!
//! EnabledAccs<Dim,Size> = tuple<Acc1<Dim,Size>, ..., AccN<Dim,Size>>
//#############################################################################
template<
    typename TTestParamSet>
struct ConvertTestParamSetToAccImpl
{
    using type =
        typename alpaka::test::acc::EnabledAccs<
            typename std::tuple_element<0, TTestParamSet>::type,
            typename std::tuple_element<1, TTestParamSet>::type
        >;
};

template<
    typename TTestParamSet>
using ConvertTestParamSetToAcc = typename ConvertTestParamSetToAccImpl<TTestParamSet>::type;

//#############################################################################
//! A std::tuple containing std::tuple with fully specialized accelerators.
//!
//! TestEnabledAccs =
//!     tuple<
//!         tuple<Acc1<Dim1,Size1>, ..., AccN<Dim1,Size1>>,
//!         tuple<Acc1<Dim2,Size1>, ..., AccN<Dim2,Size1>>,
//!         ...,
//!         tuple<Acc1<DimN,SizeN>, ..., AccN<DimN,SizeN>>>
//#############################################################################
using TestEnabledAccs =
    alpaka::meta::Transform<
        TestParamSets,
        ConvertTestParamSetToAcc
    >;

//#############################################################################
//! A std::tuple containing fully specialized accelerators.
//!
//! TestAccs =
//!     tuple<
//!         Acc1<Dim1,Size1>, ..., AccN<Dim1,Size1>,
//!         Acc1<Dim2,Size1>, ..., AccN<Dim2,Size1>,
//!         ...,
//!         Acc1<DimN,SizeN>, ..., AccN<DimN,SizeN>>
//#############################################################################
using TestAccs =
    alpaka::meta::Apply<
        TestEnabledAccs,
        alpaka::meta::Concatenate
    >;

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    getAccName,
    TAcc,
    TestAccs)
{
    std::cout << alpaka::acc::getAccName<TAcc>() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
