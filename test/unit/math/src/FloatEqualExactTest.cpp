/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/Unused.hpp>
#include <alpaka/math/FloatEqualExact.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

class FloatEqualExactTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        alpaka::ignore_unused(acc);

        // Store the comparison result in a separate variable so that the function call is outside ALPAKA_CHECK.
        // In case ALPAKA_CHECK were ever somehow modified to silence the warning by itself.
        bool test_value = false;

        float float_value = -1.0f;
        test_value = alpaka::math::floatEqualExactNoWarning(float_value, -1.0f);
        ALPAKA_CHECK(*success, test_value);

        double double_value = -1.0;
        test_value = alpaka::math::floatEqualExactNoWarning(double_value, -1.0);
        ALPAKA_CHECK(*success, test_value);
    }
};

TEMPLATE_LIST_TEST_CASE("floatEqualExactTest", "[math]", alpaka::test::TestAccs)
{
    // Host tests

    // Store the comparison result in a separate variable so that the function call is outside REQUIRE.
    // In case REQUIRE were ever somehow modified to silence the warning by itself.
    bool test_value = false;

    float float_value = -1.0;
    test_value = alpaka::math::floatEqualExactNoWarning(float_value, -1.0f);
    REQUIRE(test_value);

    double double_value = -1.0;
    test_value = alpaka::math::floatEqualExactNoWarning(double_value, -1.0);
    REQUIRE(test_value);

    // Device tests

    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    FloatEqualExactTestKernel kernel_float;
    REQUIRE(fixture(kernel_float));
}
