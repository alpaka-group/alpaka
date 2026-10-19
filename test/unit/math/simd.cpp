/* Copyright 2026 alpaka contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#include "executeOnComputeDevice.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace
{
    template<std::floating_point T, uint32_t T_width, bool T_emulated>
    struct UnarySimdKernel
    {
        ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const&, alpaka::concepts::IMdSpan auto success)
            const
        {
            using Simd = std::conditional_t<
                T_emulated,
                alpaka::Simd<T, T_width, alpaka::internal::EmuSimd<T, T_width>>,
                alpaka::Simd<T, T_width>>;
            auto const input = Simd{[](auto lane) { return (static_cast<T>(lane) - T{2}) * T{0.75}; }};

            auto const check = [&](auto operation, Simd const& arg)
            {
                auto const result = operation(arg);
                for(uint32_t lane = 0u; lane < T_width; ++lane)
                {
                    T const expected = operation(static_cast<T>(arg[lane]));
                    ALPAKA_CHECK(
                        success[0],
                        alpaka::math::abs(result[lane] - expected)
                            <= T{8} * std::numeric_limits<T>::epsilon() * (T{1} + alpaka::math::abs(expected)));
                }
            };

            check([](auto const& arg) { return alpaka::math::floor(arg); }, input);
            check([](auto const& arg) { return alpaka::math::ceil(arg); }, input);
            check([](auto const& arg) { return alpaka::math::round(arg); }, input);
            check([](auto const& arg) { return alpaka::math::trunc(arg); }, input);
            check([](auto const& arg) { return alpaka::math::abs(arg); }, input);
            check([](auto const& arg) { return alpaka::math::sin(arg); }, input);
            check([](auto const& arg) { return alpaka::math::exp(arg); }, input);
            check([](auto const& arg) { return alpaka::math::sqrt(arg); }, input * input + T{1});
        }
    };
} // namespace

namespace custom
{
    struct NativeFloorStorage : alpaka::internal::EmuSimd<double, 4u>
    {
        constexpr auto const& asNativeType() const
        {
            return *this;
        }
    };

    constexpr auto floor(NativeFloorStorage const& arg)
    {
        auto result = arg;
        // A distinct value verifies that native ADL wins over scalar evaluation.
        result[0] = 42.;
        return result;
    }
} // namespace custom

TEST_CASE("unary SIMD math preserves native ADL", "[math][simd][adl]")
{
    using Simd = alpaka::Simd<double, 4u, custom::NativeFloorStorage>;
    auto const result = alpaka::math::floor(Simd{});
    CHECK(result[0] == 42.);
}

using TestBackends
    = ALPAKA_TYPEOF(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors));

TEMPLATE_LIST_TEST_CASE("unary SIMD math", "[math][simd]", TestBackends)
{
    auto deviceExec = alpaka::test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = alpaka::test::getDevice(deviceExec);
    auto exec = alpaka::test::getExecutor(deviceExec);

    auto const check = [&]<std::floating_point T, uint32_t T_width, bool T_emulated>()
    { REQUIRE(alpaka::test::executeOnComputeDevice<T>(device, exec, UnarySimdKernel<T, T_width, T_emulated>{})); };

    check.template operator()<float, 1u, true>();
    check.template operator()<float, 4u, true>();
    check.template operator()<double, 1u, true>();
    check.template operator()<double, 4u, true>();
    check.template operator()<float, 1u, false>();
    check.template operator()<float, 4u, false>();
    check.template operator()<double, 1u, false>();
    check.template operator()<double, 4u, false>();
}
