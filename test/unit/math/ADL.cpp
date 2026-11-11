/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Buffer.hpp"
#include "DataGen.hpp"
#include "Defines.hpp"
#include "Functor.hpp"
#include "executeOnComputeDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/math/Complex.hpp>
#include <alpaka/meta/meta.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace alpaka;
using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

namespace custom
{
    enum Custom
    {
        absOp,
        acosOp,
        acoshOp,
        argOp,
        asinOp,
        asinhOp,
        atanOp,
        atanhOp,
        atan2Op,
        cbrtOp,
        ceilOp,
        conjOp,
        copysignOp,
        cosOp,
        coshOp,
        erfOp,
        expOp,
        floorOp,
        fmaOp,
        fmodOp,
        logOp,
        log2Op,
        log10Op,
        maxOp,
        minOp,
        powOp,
        remainderOp,
        roundOp,
        lroundOp,
        llroundOp,
        rsqrtOp,
        sinOp,
        sinhOp,
        sincosOp,
        sqrtOp,
        tanOp,
        tanhOp,
        truncOp,

        arg1 = 1024,
        arg2 = 2048,
        arg3 = 4096,

        // expected results of a custom sincos(arg1, a, b) definition using bitwise or
        expectedCustomSincosArg1Arg2 = sincosOp | arg1 | arg2,
        expectedCustomSincosArg1Arg3 = sincosOp | arg1 | arg3,
    };

    ALPAKA_FN_HOST_ACC auto abs(Custom c);

    ALPAKA_FN_HOST_ACC auto abs(Custom c)
    {
        return Custom::absOp | c;
    }

    ALPAKA_FN_HOST_ACC auto acos(Custom c);

    ALPAKA_FN_HOST_ACC auto acos(Custom c)
    {
        return Custom::acosOp | c;
    }

    ALPAKA_FN_HOST_ACC auto acosh(Custom c);

    ALPAKA_FN_HOST_ACC auto acosh(Custom c)
    {
        return Custom::acoshOp | c;
    }

    ALPAKA_FN_HOST_ACC auto arg(Custom c);

    ALPAKA_FN_HOST_ACC auto arg(Custom c)
    {
        return Custom::argOp | c;
    }

    ALPAKA_FN_HOST_ACC auto asin(Custom c);

    ALPAKA_FN_HOST_ACC auto asin(Custom c)
    {
        return Custom::asinOp | c;
    }

    ALPAKA_FN_HOST_ACC auto asinh(Custom c);

    ALPAKA_FN_HOST_ACC auto asinh(Custom c)
    {
        return Custom::asinhOp | c;
    }

    ALPAKA_FN_HOST_ACC auto atan(Custom c);

    ALPAKA_FN_HOST_ACC auto atan(Custom c)
    {
        return Custom::atanOp | c;
    }

    ALPAKA_FN_HOST_ACC auto atanh(Custom c);

    ALPAKA_FN_HOST_ACC auto atanh(Custom c)
    {
        return Custom::atanhOp | c;
    }

    ALPAKA_FN_HOST_ACC auto atan2(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto atan2(Custom a, Custom b)
    {
        return Custom::atan2Op | a | b;
    }

    ALPAKA_FN_HOST_ACC auto cbrt(Custom c);

    ALPAKA_FN_HOST_ACC auto cbrt(Custom c)
    {
        return Custom::cbrtOp | c;
    }

    ALPAKA_FN_HOST_ACC auto ceil(Custom c);

    ALPAKA_FN_HOST_ACC auto ceil(Custom c)
    {
        return Custom::ceilOp | c;
    }

    ALPAKA_FN_HOST_ACC auto conj(Custom c);

    ALPAKA_FN_HOST_ACC auto conj(Custom c)
    {
        return Custom::conjOp | c;
    }

    ALPAKA_FN_HOST_ACC auto copysign(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto copysign(Custom a, Custom b)
    {
        return Custom::copysignOp | a | b;
    }

    ALPAKA_FN_HOST_ACC auto cos(Custom c);

    ALPAKA_FN_HOST_ACC auto cos(Custom c)
    {
        return Custom::cosOp | c;
    }

    ALPAKA_FN_HOST_ACC auto cosh(Custom c);

    ALPAKA_FN_HOST_ACC auto cosh(Custom c)
    {
        return Custom::coshOp | c;
    }

    ALPAKA_FN_HOST_ACC auto erf(Custom c);

    ALPAKA_FN_HOST_ACC auto erf(Custom c)
    {
        return Custom::erfOp | c;
    }

    ALPAKA_FN_HOST_ACC auto exp(Custom c);

    ALPAKA_FN_HOST_ACC auto exp(Custom c)
    {
        return Custom::expOp | c;
    }

    ALPAKA_FN_HOST_ACC auto floor(Custom c);

    ALPAKA_FN_HOST_ACC auto floor(Custom c)
    {
        return Custom::floorOp | c;
    }

    ALPAKA_FN_HOST_ACC auto fma(Custom a, Custom b, Custom c);

    ALPAKA_FN_HOST_ACC auto fma(Custom a, Custom b, Custom c)
    {
        return Custom::fmaOp | a | b | c;
    }

    ALPAKA_FN_HOST_ACC auto fmod(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto fmod(Custom a, Custom b)
    {
        return Custom::fmodOp | a | b;
    }

    ALPAKA_FN_HOST_ACC auto log(Custom c);

    ALPAKA_FN_HOST_ACC auto log(Custom c)
    {
        return Custom::logOp | c;
    }

    ALPAKA_FN_HOST_ACC auto log2(Custom c);

    ALPAKA_FN_HOST_ACC auto log2(Custom c)
    {
        return Custom::log2Op | c;
    }

    ALPAKA_FN_HOST_ACC auto log10(Custom c);

    ALPAKA_FN_HOST_ACC auto log10(Custom c)
    {
        return Custom::log10Op | c;
    }

    ALPAKA_FN_HOST_ACC auto max(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto max(Custom a, Custom b)
    {
        return Custom::maxOp | a | b;
    }

    ALPAKA_FN_HOST_ACC auto min(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto min(Custom a, Custom b)
    {
        return Custom::minOp | a | b;
    }

    ALPAKA_FN_HOST_ACC auto pow(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto pow(Custom a, Custom b)
    {
        return Custom::powOp | a | b;
    }

    ALPAKA_FN_HOST_ACC auto remainder(Custom a, Custom b);

    ALPAKA_FN_HOST_ACC auto remainder(Custom a, Custom b)
    {
        return Custom::remainderOp | a | b;
    }

    ALPAKA_FN_HOST_ACC auto round(Custom c);

    ALPAKA_FN_HOST_ACC auto round(Custom c)
    {
        return Custom::roundOp | c;
    }

    ALPAKA_FN_HOST_ACC auto lround(Custom c);

    ALPAKA_FN_HOST_ACC auto lround(Custom c)
    {
        return Custom::lroundOp | c;
    }

    ALPAKA_FN_HOST_ACC auto llround(Custom c);

    ALPAKA_FN_HOST_ACC auto llround(Custom c)
    {
        return Custom::llroundOp | c;
    }

    ALPAKA_FN_HOST_ACC auto rsqrt(Custom c);

    ALPAKA_FN_HOST_ACC auto rsqrt(Custom c)
    {
        return Custom::rsqrtOp | c;
    }

    ALPAKA_FN_HOST_ACC auto sin(Custom c);

    ALPAKA_FN_HOST_ACC auto sin(Custom c)
    {
        return Custom::sinOp | c;
    }

    ALPAKA_FN_HOST_ACC auto sinh(Custom c);

    ALPAKA_FN_HOST_ACC auto sinh(Custom c)
    {
        return Custom::sinhOp | c;
    }

    ALPAKA_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b);

    ALPAKA_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b)
    {
        a = static_cast<Custom>(Custom::sincosOp | c | Custom::arg2);
        b = static_cast<Custom>(Custom::sincosOp | c | Custom::arg3);
    }

    ALPAKA_FN_HOST_ACC auto sqrt(Custom c);

    ALPAKA_FN_HOST_ACC auto sqrt(Custom c)
    {
        return Custom::sqrtOp | c;
    }

    ALPAKA_FN_HOST_ACC auto tan(Custom c);

    ALPAKA_FN_HOST_ACC auto tan(Custom c)
    {
        return Custom::tanOp | c;
    }

    ALPAKA_FN_HOST_ACC auto tanh(Custom c);

    ALPAKA_FN_HOST_ACC auto tanh(Custom c)
    {
        return Custom::tanhOp | c;
    }

    ALPAKA_FN_HOST_ACC auto trunc(Custom c);

    ALPAKA_FN_HOST_ACC auto trunc(Custom c)
    {
        return Custom::truncOp | c;
    }
} // namespace custom

struct AdlKernel
{
    template<typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const&, alpaka::concepts::IMdSpan auto success) const noexcept
    {
        using custom::Custom;

        ALPAKA_CHECK(success[0], alpaka::math::abs(Custom::arg1) == (Custom::absOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::acos(Custom::arg1) == (Custom::acosOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::acosh(Custom::arg1) == (Custom::acoshOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::arg(Custom::arg1) == (Custom::argOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::asin(Custom::arg1) == (Custom::asinOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::asinh(Custom::arg1) == (Custom::asinhOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::atan(Custom::arg1) == (Custom::atanOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::atanh(Custom::arg1) == (Custom::atanhOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::cbrt(Custom::arg1) == (Custom::cbrtOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::ceil(Custom::arg1) == (Custom::ceilOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::conj(Custom::arg1) == (Custom::conjOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::cos(Custom::arg1) == (Custom::cosOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::cosh(Custom::arg1) == (Custom::coshOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::erf(Custom::arg1) == (Custom::erfOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::exp(Custom::arg1) == (Custom::expOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::floor(Custom::arg1) == (Custom::floorOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::log(Custom::arg1) == (Custom::logOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::log2(Custom::arg1) == (Custom::log2Op | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::log10(Custom::arg1) == (Custom::log10Op | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::round(Custom::arg1) == (Custom::roundOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::lround(Custom::arg1) == (Custom::lroundOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::llround(Custom::arg1) == (Custom::llroundOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::rsqrt(Custom::arg1) == (Custom::rsqrtOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::sin(Custom::arg1) == (Custom::sinOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::sinh(Custom::arg1) == (Custom::sinhOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::sqrt(Custom::arg1) == (Custom::sqrtOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::tan(Custom::arg1) == (Custom::tanOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::tanh(Custom::arg1) == (Custom::tanhOp | Custom::arg1));
        ALPAKA_CHECK(success[0], alpaka::math::trunc(Custom::arg1) == (Custom::truncOp | Custom::arg1));

        ALPAKA_CHECK(
            success[0],
            alpaka::math::atan2(Custom::arg1, Custom::arg2) == (Custom::atan2Op | Custom::arg1 | Custom::arg2));
        ALPAKA_CHECK(
            success[0],
            alpaka::math::copysign(Custom::arg1, Custom::arg2) == (Custom::copysignOp | Custom::arg1 | Custom::arg2));

        ALPAKA_CHECK(
            success[0],
            alpaka::math::fmod(Custom::arg1, Custom::arg2) == (Custom::fmodOp | Custom::arg1 | Custom::arg2));

        ALPAKA_CHECK(
            success[0],
            alpaka::math::max(Custom::arg1, Custom::arg2) == (Custom::maxOp | Custom::arg1 | Custom::arg2));
        ALPAKA_CHECK(
            success[0],
            alpaka::math::min(Custom::arg1, Custom::arg2) == (Custom::minOp | Custom::arg1 | Custom::arg2));
        ALPAKA_CHECK(
            success[0],
            alpaka::math::pow(Custom::arg1, Custom::arg2) == (Custom::powOp | Custom::arg1 | Custom::arg2));

        ALPAKA_CHECK(
            success[0],
            alpaka::math::remainder(Custom::arg1, Custom::arg2)
                == (Custom::remainderOp | Custom::arg1 | Custom::arg2));

        ALPAKA_CHECK(
            success[0],
            alpaka::math::fma(Custom::arg1, Custom::arg2, Custom::arg3)
                == (Custom::fmaOp | Custom::arg1 | Custom::arg2 | Custom::arg3));

        Custom a, b;
        alpaka::math::sincos(Custom::arg1, a, b);

        ALPAKA_CHECK(success[0], a == Custom::expectedCustomSincosArg1Arg2);
        ALPAKA_CHECK(success[0], b == Custom::expectedCustomSincosArg1Arg3);
    }
};

TEMPLATE_LIST_TEST_CASE("mathOps", "[math] [operator] [adl]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto testResult = alpaka::test::executeOnComputeDevice(device, exec, AdlKernel{});
    REQUIRE(testResult);
}
