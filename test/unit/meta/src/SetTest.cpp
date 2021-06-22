/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Set.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("isSetTrue", "[meta]")
{
    using IsSetInput = std::tuple<int, float, long>;

    constexpr bool is_set_result = alpaka::meta::IsSet<IsSetInput>::value;

    constexpr bool is_set_reference = true;

    static_assert(is_set_reference == is_set_result, "alpaka::meta::IsSet failed!");
}

TEST_CASE("isSetFalse", "[meta]")
{
    using IsSetInput = std::tuple<int, float, int>;

    constexpr bool is_set_result = alpaka::meta::IsSet<IsSetInput>::value;

    constexpr bool is_set_reference = false;

    static_assert(is_set_reference == is_set_result, "alpaka::meta::IsSet failed!");
}
