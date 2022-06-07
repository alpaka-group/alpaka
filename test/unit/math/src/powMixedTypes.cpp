/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/math/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <type_traits>

// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<typename TAcc, typename FP>
ALPAKA_FN_ACC auto almost_equal(TAcc const& acc, FP x, FP y, int ulp)
    -> std::enable_if_t<!std::numeric_limits<FP>::is_integer, bool>
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return alpaka::math::abs(acc, x - y)
        <= std::numeric_limits<FP>::epsilon() * alpaka::math::abs(acc, x + y) * static_cast<FP>(ulp)
        // unless the result is subnormal
        || alpaka::math::abs(acc, x - y) < std::numeric_limits<FP>::min();
}

//! Version for alpaka::Complex
template<typename TAcc, typename FP>
ALPAKA_FN_ACC bool almost_equal(TAcc const& acc, alpaka::Complex<FP> x, alpaka::Complex<FP> y, int ulp)
{
    return almost_equal(acc, x.real(), y.real(), ulp) && almost_equal(acc, x.imag(), y.imag(), ulp);
}

template<typename TExpected>
class PowMixedTypesTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TArg1, typename TArg2>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, TArg1 const arg1, TArg2 const arg2) const -> void
    {
        auto expected = alpaka::math::pow(acc, TExpected{arg1}, TExpected{arg2});
        auto actual = alpaka::math::pow(acc, arg1, arg2);
        ALPAKA_CHECK(*success, almost_equal(acc, expected, actual, 1));
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("powMixedTypes", "[powMixedTypes]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PowMixedTypesTestKernel<float> kernelFloat;
    PowMixedTypesTestKernel<double> kernelDouble;
    PowMixedTypesTestKernel<alpaka::Complex<float>> kernelComplexFloat;
    PowMixedTypesTestKernel<alpaka::Complex<double>> kernelComplexDouble;

    float const floatArg = 0.35f;
    double const doubleArg = 0.24;
    alpaka::Complex<float> floatComplexArg{0.35f, -0.24f};
    alpaka::Complex<double> doubleComplexArg{0.35, -0.24};

    // all combinations of pow(real, real)
    REQUIRE(fixture(kernelFloat, floatArg, floatArg));
    REQUIRE(fixture(kernelDouble, floatArg, doubleArg));
    REQUIRE(fixture(kernelDouble, doubleArg, floatArg));
    REQUIRE(fixture(kernelDouble, doubleArg, doubleArg));

    // all combinations of pow(real, complex)
    REQUIRE(fixture(kernelComplexFloat, floatArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, floatArg, doubleComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleArg, doubleComplexArg));

    // all combinations of pow(complex, real)
    REQUIRE(fixture(kernelComplexFloat, floatComplexArg, floatArg));
    REQUIRE(fixture(kernelComplexDouble, floatComplexArg, doubleArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, floatArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, doubleArg));

    // all combinations of pow(complex, complex)
    REQUIRE(fixture(kernelComplexFloat, floatComplexArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, floatComplexArg, doubleComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, floatComplexArg));
    REQUIRE(fixture(kernelComplexDouble, doubleComplexArg, doubleComplexArg));
}
