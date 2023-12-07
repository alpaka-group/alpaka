/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/Apply.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

template<typename... T>
struct TypeList
{
};

TEST_CASE("apply", "[meta]")
{
    using ApplyInput = std::tuple<int, float, long>;

    using ApplyResult = alpaka::meta::Apply<ApplyInput, TypeList>;

    using ApplyReference = TypeList<int, float, long>;

    static_assert(std::is_same_v<ApplyReference, ApplyResult>, "alpaka::meta::Apply failed!");
}
