/* Copyright 2025 René Widera, Mehmet Yusufoglu, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/api/generic.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <tuple>

template<typename T>
static void verifyIsSpecial()
{
    namespace genericMath = alpaka::internal::generic::math;

    auto const quietNaN = std::numeric_limits<T>::quiet_NaN();
    auto const posInf = std::numeric_limits<T>::infinity();
    auto const negInf = -posInf;
    auto const finiteVal = static_cast<T>(42.5);

    CHECK(genericMath::isnan(quietNaN));
    CHECK_FALSE(genericMath::isnan(finiteVal));

    if constexpr(std::numeric_limits<T>::has_signaling_NaN)
    {
        auto const signalingNaN = std::numeric_limits<T>::signaling_NaN();
        CHECK(genericMath::isnan(signalingNaN));
    }

    CHECK(genericMath::isinf(posInf));
    CHECK(genericMath::isinf(negInf));
    CHECK_FALSE(genericMath::isinf(finiteVal));

    CHECK(genericMath::isfinite(finiteVal));
    CHECK(genericMath::isfinite(static_cast<T>(0)));
    CHECK_FALSE(genericMath::isfinite(posInf));
    CHECK_FALSE(genericMath::isfinite(negInf));
    CHECK_FALSE(genericMath::isfinite(quietNaN));
}

using TestValueTypes = std::tuple<float, double>;

// Regression guard to ensure ieee helper stays stable for each floating type.
TEMPLATE_LIST_TEST_CASE("generic ieee helpers detect special values", "[math][generic][ieee]", TestValueTypes)
{
    verifyIsSpecial<TestType>();
}
