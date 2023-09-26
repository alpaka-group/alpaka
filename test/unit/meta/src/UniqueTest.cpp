/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/meta/Unique.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("uniqueWithDuplicate", "[meta]")
{
    using UniqueInput = std::tuple<int, float, int, float, float, int>;

    using UniqueResult = alpaka::meta::Unique<UniqueInput>;

    using UniqueReference = std::tuple<int, float>;

    static_assert(std::is_same_v<UniqueReference, UniqueResult>, "alpaka::meta::Unique failed!");
}

TEST_CASE("uniqueWithoutDuplicate", "[meta]")
{
    using UniqueInput = std::tuple<int, float, double>;

    using UniqueResult = alpaka::meta::Unique<UniqueInput>;

    using UniqueReference = UniqueInput;

    static_assert(std::is_same_v<UniqueReference, UniqueResult>, "alpaka::meta::Unique failed!");
}
