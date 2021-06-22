/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/IsStrictBase.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

class A
{
};
class B : A
{
};
class C
{
};

TEST_CASE("isStrictBaseTrue", "[meta]")
{
    constexpr bool is_strict_base_result = alpaka::meta::IsStrictBase<A, B>::value;

    constexpr bool is_strict_base_reference = true;

    static_assert(is_strict_base_reference == is_strict_base_result, "alpaka::meta::IsStrictBase failed!");
}

TEST_CASE("isStrictBaseIdentity", "[meta]")
{
    constexpr bool is_strict_base_result = alpaka::meta::IsStrictBase<A, A>::value;

    constexpr bool is_strict_base_reference = false;

    static_assert(is_strict_base_reference == is_strict_base_result, "alpaka::meta::IsStrictBase failed!");
}

TEST_CASE("isStrictBaseNoInheritance", "[meta]")
{
    constexpr bool is_strict_base_result = alpaka::meta::IsStrictBase<A, C>::value;

    constexpr bool is_strict_base_reference = false;

    static_assert(is_strict_base_reference == is_strict_base_result, "alpaka::meta::IsStrictBase failed!");
}

TEST_CASE("isStrictBaseWrongOrder", "[meta]")
{
    constexpr bool is_strict_base_result = alpaka::meta::IsStrictBase<B, A>::value;

    constexpr bool is_strict_base_reference = false;

    static_assert(is_strict_base_reference == is_strict_base_result, "alpaka::meta::IsStrictBase failed!");
}
