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

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>


//-----------------------------------------------------------------------------
TEST_CASE("filter", "[meta]")
{
    using FilterInput =
        std::tuple<
            int,
            float,
            long>;

    using FilterResult =
        alpaka::meta::Filter<
            FilterInput,
            std::is_integral
        >;

    using FilterReference =
        std::tuple<
            int,
            long>;

    static_assert(
        std::is_same<
            FilterReference,
            FilterResult
        >::value,
        "alpaka::meta::Filter failed!");
}
