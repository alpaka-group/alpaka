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

#include <string>
#include <tuple>
#include <type_traits>


//-----------------------------------------------------------------------------
TEST_CASE("concatenate", "[meta]")
{
    using TestTuple1 =
        std::tuple<
            float,
            int,
            std::tuple<double, unsigned long>>;

    using TestTuple2 =
        std::tuple<
            bool,
            std::string>;

    using ConcatenateResult =
        alpaka::meta::Concatenate<
            TestTuple1,
            TestTuple2
        >;

    using ConcatenateReference =
        std::tuple<
            float,
            int,
            std::tuple<double, unsigned long>,
            bool,
            std::string>;

    static_assert(
        std::is_same<
            ConcatenateReference,
            ConcatenateResult
        >::value,
        "alpaka::meta::Concatenate failed!");
}
