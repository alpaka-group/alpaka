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

#include <alpaka/meta/Concatenate.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <tuple>
#include <type_traits>

TEST_CASE("concatenate", "[meta]")
{
    using TestTuple1 = std::tuple<float, int, std::tuple<double, unsigned long>>;

    using TestTuple2 = std::tuple<bool, std::string>;

    using ConcatenateResult = alpaka::meta::Concatenate<TestTuple1, TestTuple2>;

    using ConcatenateReference = std::tuple<float, int, std::tuple<double, unsigned long>, bool, std::string>;

    static_assert(std::is_same_v<ConcatenateReference, ConcatenateResult>, "alpaka::meta::Concatenate failed!");
}
