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

template<
    typename T>
using AddConst = T const;

//-----------------------------------------------------------------------------
TEST_CASE("transform", "[meta]")
{
    using TransformInput =
        std::tuple<
            int,
            float,
            long>;

    using TransformResult =
        alpaka::meta::Transform<
            TransformInput,
            AddConst
        >;

    using TransformReference =
        std::tuple<
            int const,
            float const,
            long const>;

    static_assert(
        std::is_same<
            TransformReference,
            TransformResult
        >::value,
        "alpaka::meta::Transform failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("transformVariadic", "[meta]")
{
    using TransformInput =
        std::tuple<
            int,
            float,
            long>;

    using TransformResult =
        alpaka::meta::Transform<
            TransformInput,
            std::tuple
        >;

    using TransformReference =
        std::tuple<
            std::tuple<int>,
            std::tuple<float>,
            std::tuple<long>>;

    static_assert(
        std::is_same<
            TransformReference,
            TransformResult
        >::value,
        "alpaka::meta::Transform failed!");
}
