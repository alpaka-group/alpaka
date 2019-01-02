/**
 * \file
 * Copyright 2015-2019 Benjamin Worpitz
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

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>


//-----------------------------------------------------------------------------
TEST_CASE("cartesianProduct", "[meta]")
{
    using TestDims =
        std::tuple<
            alpaka::dim::DimInt<1u>,
            alpaka::dim::DimInt<2u>,
            alpaka::dim::DimInt<3u>>;

    using TestIdxs =
        std::tuple<
            std::size_t,
            std::int64_t>;

    using CartesianProductResult =
        alpaka::meta::CartesianProduct<
            std::tuple,
            TestDims,
            TestIdxs
        >;

    using CartesianProductReference =
        std::tuple<
            std::tuple<alpaka::dim::DimInt<1u>, std::size_t>,
            std::tuple<alpaka::dim::DimInt<2u>, std::size_t>,
            std::tuple<alpaka::dim::DimInt<3u>, std::size_t>,
            std::tuple<alpaka::dim::DimInt<1u>, std::int64_t>,
            std::tuple<alpaka::dim::DimInt<2u>, std::int64_t>,
            std::tuple<alpaka::dim::DimInt<3u>, std::int64_t>>;

    static_assert(
        std::is_same<
            CartesianProductReference,
            CartesianProductResult
        >::value,
        "alpaka::meta::CartesianProduct failed!");
}
