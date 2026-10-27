/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Functor.hpp"
#include "TestTemplate.hpp"
#include "alpaka/math/Complex.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/meta.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <tuple>
#include <type_traits>

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

// This file only has unit tests for complex numbers in order to split the tests between object files and save compiler
// memory. For the same reason single- and double-precision are done separately and not wrapped into a common template.
using FunctorsComplex = alpaka::meta::Concatenate<mathtest::UnaryFunctorsComplex, mathtest::BinaryFunctorsComplex>;
using TestAccFunctorTuplesComplex = alpaka::meta::CartesianProduct<std::tuple, TestBackends, FunctorsComplex>;

TEMPLATE_LIST_TEST_CASE("mathOpsComplexFloat", "[math] [operator]", TestAccFunctorTuplesComplex)
{
    // Same as "mathOpsFloat" template test, but for complex float. See detailed explanation there.
    using Backend = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;
    auto cfg = Backend::makeDict();
    auto testTemplate = mathtest::TestTemplate<Functor>{};
    testTemplate.template operator()<alpaka::math::Complex<float>>(cfg);
}

TEST_CASE("mathArrayOrientedComplexFloat", "[array-oriented]")
{
    // Ensure that our implementation matches the behaviour of std::complex with regard to array-oriented access.
    // See https://en.cppreference.com/w/cpp/numeric/complex - Array-oriented access - for more information.
    auto const complexAlpaka = alpaka::math::Complex<float>{42.f, 42.f};
    auto const complexStd = std::complex<float>{42.f, 42.f};

    auto const realAlpaka = reinterpret_cast<float const(&)[2]>(complexAlpaka)[0];
    auto const realStd = reinterpret_cast<float const(&)[2]>(complexStd)[0];
    REQUIRE(alpaka::math::floatEqualExactNoWarning(realAlpaka, realStd));

    auto const imagAlpaka = reinterpret_cast<float const(&)[2]>(complexAlpaka)[1];
    auto const imagStd = reinterpret_cast<float const(&)[2]>(complexStd)[1];
    REQUIRE(alpaka::math::floatEqualExactNoWarning(imagAlpaka, imagStd));
}

TEST_CASE("mathPaddingComplexFloat", "[padding]")
{
    // Ensure that we don't accidentally introduce padding
    STATIC_REQUIRE(sizeof(alpaka::math::Complex<float>) == 2 * sizeof(float));
}
