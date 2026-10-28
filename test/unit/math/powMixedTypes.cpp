/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Defines.hpp"
#include "executeOnComputeDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/meta.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace alpaka;

//! Convert the given real or complex input type to the given real or complex output type avoiding warnings.
//! This conversion is surprisingly tricky to do in a way that no compiler complains.
//! In principle it could be accomplished by a constexpr function, but in practice that turned out not possible.
//! The general implementation does direct initialization, works for pairs of types supporting it.
template<typename T_Input, typename T_Output, typename T_Sfinae = void>
struct Convert
{
    ALPAKA_FN_ACC auto operator()(T_Input const arg) const
    {
        return T_Output{arg};
    }
};

//! Specialization for real -> real conversion
template<typename T_Input, typename T_Output>
struct Convert<T_Input, T_Output, std::enable_if_t<std::is_floating_point_v<T_Output>>>
{
    ALPAKA_FN_ACC auto operator()(T_Input const arg) const
    {
        return static_cast<T_Output>(arg);
    }
};

//! Specialization for real -> complex conversion
template<typename T_Input, typename T_OutputValueType>
struct Convert<T_Input, alpaka::math::Complex<T_OutputValueType>, std::enable_if_t<std::is_floating_point_v<T_Input>>>
{
    ALPAKA_FN_ACC auto operator()(T_Input const arg) const
    {
        return alpaka::math::Complex<T_OutputValueType>{static_cast<T_OutputValueType>(arg)};
    }
};

template<typename T_Expected>
struct PowMixedTypesTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T_Acc, typename T_Arg1, typename T_Arg2>
    ALPAKA_FN_ACC auto operator()(
        T_Acc const&,
        concepts::IMdSpan<bool> auto success,
        T_Arg1 const arg1,
        T_Arg2 const arg2) const -> void
    {
        auto expected = alpaka::math::pow(Convert<T_Arg1, T_Expected>{}(arg1), Convert<T_Arg2, T_Expected>{}(arg2));
        auto actual = alpaka::math::pow(arg1, arg2);
        ALPAKA_CHECK(success[0], mathtest::almost_equal(expected, actual, 1));
    }
};

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("powMixedTypes", "[powMixedTypes]", TestBackends)
{
    auto cfg = TestType::makeDict();

    PowMixedTypesTestKernel<float> kernelFloat;
    PowMixedTypesTestKernel<double> kernelDouble;
    PowMixedTypesTestKernel<alpaka::math::Complex<float>> kernelComplexFloat;
    PowMixedTypesTestKernel<alpaka::math::Complex<double>> kernelComplexDouble;

    float const floatArg = 0.35f;
    double const doubleArg = 0.24;
    alpaka::math::Complex<float> floatComplexArg{0.35f, -0.24f};
    alpaka::math::Complex<double> doubleComplexArg{0.35, -0.24};

    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    // all combinations of pow(real, real)
    REQUIRE(alpaka::test::executeOnComputeDevice(device, exec, kernelFloat, floatArg, floatArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(device, exec, kernelDouble, floatArg, doubleArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(device, exec, kernelDouble, doubleArg, floatArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(device, exec, kernelDouble, doubleArg, doubleArg));

    // all combinations of pow(real, complex)
    REQUIRE(alpaka::test::executeOnComputeDevice(device, exec, kernelComplexFloat, floatArg, floatComplexArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(device, exec, kernelComplexDouble, floatArg, doubleComplexArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(device, exec, kernelComplexDouble, doubleArg, floatComplexArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(device, exec, kernelComplexDouble, doubleArg, doubleComplexArg));

    // all combinations of pow(complex, real)
    REQUIRE(alpaka::test::executeOnComputeDevice(device, exec, kernelComplexFloat, floatComplexArg, floatArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(device, exec, kernelComplexDouble, floatComplexArg, doubleArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(device, exec, kernelComplexDouble, doubleComplexArg, floatArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(device, exec, kernelComplexDouble, doubleComplexArg, doubleArg));

    // all combinations of pow(complex, complex)
    REQUIRE(alpaka::test::executeOnComputeDevice(device, exec, kernelComplexFloat, floatComplexArg, floatComplexArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(
            device,
            exec,
            kernelComplexDouble,
            floatComplexArg,
            doubleComplexArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(
            device,
            exec,
            kernelComplexDouble,
            doubleComplexArg,
            floatComplexArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(
            device,
            exec,
            kernelComplexDouble,
            doubleComplexArg,
            doubleComplexArg));
}
