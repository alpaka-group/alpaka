/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/Set.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("isSetTrue", "[meta]")
{
    using IsSetInput = std::tuple<int, float, long>;

    constexpr bool IsSetResult = alpaka::meta::IsSet<IsSetInput>::value;

    constexpr bool IsSetReference = true;

    static_assert(IsSetReference == IsSetResult, "alpaka::meta::IsSet failed!");
}

TEST_CASE("isSetFalse", "[meta]")
{
    using IsSetInput = std::tuple<int, float, int>;

    constexpr bool IsSetResult = alpaka::meta::IsSet<IsSetInput>::value;

    constexpr bool IsSetReference = false;

    static_assert(IsSetReference == IsSetResult, "alpaka::meta::IsSet failed!");
}
