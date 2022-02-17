/* Copyright 2021 Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/TypeListOps.hpp>

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("front", "[meta]")
{
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<int>>, int>);
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<int, int>>, int>);
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<float, int>>, float>);
    STATIC_REQUIRE(std::is_same_v<alpaka::meta::Front<std::tuple<short, int, double, float, float>>, short>);
}

TEST_CASE("contains", "[meta]")
{
    STATIC_REQUIRE(!alpaka::meta::Contains<std::tuple<>, int>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<int>, int>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<short, int, double, float>, short>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<short, int, double, float>, double>::value);
    STATIC_REQUIRE(alpaka::meta::Contains<std::tuple<short, int, double, float>, float>::value);
    STATIC_REQUIRE(!alpaka::meta::Contains<std::tuple<short, int, double, float>, char>::value);
}
