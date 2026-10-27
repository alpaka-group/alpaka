/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

/** @file
 *
 *  This file is testing simd where expression
 */

TEST_CASE("simd where", "[simd vector]")
{
    using namespace alpaka;

    Simd<double, 8> a{1, 2, 3, 4, 5, 6, 7, 8};
    Simd<double, 8> b{8, 7, 6, 5, 4, 3, 2, 1};

    auto m = a > b;
    auto mReference = makeSimdMask<double>(false, false, false, false, true, true, true, true);
    CHECK((m == mReference).reduce(std::logical_and{}));

    where(m, a) = b;
    Simd<double, 8> a0Reference{1, 2, 3, 4, 4, 3, 2, 1};
    CHECK((a == a0Reference).reduce(std::logical_and{}));

    where(m, a) += b;
    Simd<double, 8> a1Reference{1, 2, 3, 4, 8, 6, 4, 2};
    CHECK((a == a1Reference).reduce(std::logical_and{}));

    where(b >= 5., b) = 42.;
    Simd<double, 8> b0Reference{42, 42, 42, 42, 4, 3, 2, 1};
    CHECK((b == b0Reference).reduce(std::logical_and{}));
    where(b >= 5., b) += 1.;
    Simd<double, 8> b1Reference{43, 43, 43, 43, 4, 3, 2, 1};
    CHECK((b == b1Reference).reduce(std::logical_and{}));

    // test upcast of scalar values
    where(b >= float{4}, b) = float{43};
    Simd<double, 8> b2Reference{43, 43, 43, 43, 43, 3, 2, 1};
    CHECK((b == b2Reference).reduce(std::logical_and{}));

    where(b < float{4}, b) -= float{3};
    Simd<double, 8> b3Reference{43, 43, 43, 43, 43, 0, -1, -2};
    CHECK((b == b3Reference).reduce(std::logical_and{}));

    Simd<double, 8> c{32, 16, 8, 4, 2, 1, 9, 27};
    where(c >= 8., c) /= 2.;
    Simd<double, 8> c0Reference{16, 8, 4, 4, 2, 1, 4.5, 13.5};
    CHECK((c == c0Reference).reduce(std::logical_and{}));

    Simd<double, 8> d{81, 64, 49, 36, 25, 16, 9, 4};
    Simd<double, 8> e{9, 8, 7, 6, 5, 4, 3, 2};
    where(d >= 25., d) /= e;
    Simd<double, 8> d0Reference{9, 8, 7, 6, 5, 16, 9, 4};
    CHECK((d == d0Reference).reduce(std::logical_and{}));
}
