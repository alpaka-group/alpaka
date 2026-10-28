/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Defines.hpp"
#include "executeOnComputeDevice.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>


using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

struct SinCosTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename T_Acc, typename T_Fp>
    ALPAKA_FN_ACC auto operator()(T_Acc const&, concepts::IMdSpan<bool> auto success, T_Fp const arg) const -> void
    {
        // if arg is hardcoded then compiler can optimize it out
        // (PTX kernel (float) was just empty)
        T_Fp checkSin = alpaka::math::sin(arg);
        T_Fp checkCos = alpaka::math::cos(arg);
        auto resultSin = T_Fp{0};
        auto resultCos = T_Fp{0};
        math::sincos(arg, resultSin, resultCos);
        ALPAKA_CHECK(
            success[0],
            mathtest::almost_equal(resultSin, checkSin, 1) && mathtest::almost_equal(resultCos, checkCos, 1));
    }
};

TEMPLATE_LIST_TEST_CASE("sincos", "[sincos]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    SinCosTestKernel kernel;

    REQUIRE(alpaka::test::executeOnComputeDevice(device, exec, kernel, 0.42f)); // float
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(device, exec, kernel, 0.42)); // double
    REQUIRE(
        alpaka::test::executeOnComputeDevice(
            device,
            exec,
            kernel,
            alpaka::math::Complex<float>{0.35f, -0.24f})); // complex
                                                           // float
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(
            device,
            exec,
            kernel,
            alpaka::math::Complex<double>{0.35, -0.24})); // complex double
}
