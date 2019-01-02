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


class A {};
class B : A {};
class C {};

//-----------------------------------------------------------------------------
TEST_CASE("isStrictBaseTrue", "[meta]")
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            A, B
        >::value;

    constexpr bool IsStrictBaseReference =
        true;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("isStrictBaseIdentity", "[meta]")
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            A, A
        >::value;

    constexpr bool IsStrictBaseReference =
        false;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("isStrictBaseNoInheritance", "[meta]")
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            A, C
        >::value;

    constexpr bool IsStrictBaseReference =
        false;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}

//-----------------------------------------------------------------------------
TEST_CASE("isStrictBaseWrongOrder", "[meta]")
{
    constexpr bool IsStrictBaseResult =
        alpaka::meta::IsStrictBase<
            B, A
        >::value;

    constexpr bool IsStrictBaseReference =
        false;

    static_assert(
        IsStrictBaseReference == IsStrictBaseResult,
        "alpaka::meta::IsStrictBase failed!");
}
