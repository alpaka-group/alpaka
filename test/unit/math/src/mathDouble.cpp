/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Sergei Bastrakov <s.bastrakov@hzdr.de>
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jakob Krude <jakob.krude@hotmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Functor.hpp"
#include "TestTemplate.hpp"

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

// This file only has unit tests for real numbers in order to split the tests between object files
using FunctorsReal = alpaka::meta::Concatenate<
    alpaka::test::unit::math::UnaryFunctorsReal,
    alpaka::test::unit::math::BinaryFunctorsReal,
    alpaka::test::unit::math::TernaryFunctorsReal>;
using TestAccFunctorTuplesReal = alpaka::meta::CartesianProduct<std::tuple, TestAccs, FunctorsReal>;

TEMPLATE_LIST_TEST_CASE("mathOpsDouble", "[math] [operator]", TestAccFunctorTuplesReal)
{
    // Same as "mathOpsFloat" template test, but for double. See detailed explanation there.
    using Acc = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;
    auto testTemplate = TestTemplate<Acc, Functor>{};
    testTemplate.template operator()<double>();
}
