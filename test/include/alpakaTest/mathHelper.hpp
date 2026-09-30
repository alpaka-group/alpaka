/* Copyright 2026 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

namespace alpaka::test
{
    //! Approximate comparison of real numbers
    template<typename T>
    constexpr bool isApproxEqual(T const& a, T const& b)
    {
        if constexpr(std::is_floating_point_v<T>)
        {
            // NaN is not equal to anything, including itself, so a NaN produced by
            // both the alpaka and the std:: reference implementation would be
            // reported as a mismatch. This is a legitimate result for some inputs
            // (e.g. an infinite operand for remainders or modulo), therefore two
            // NaNs are treated as equal here.
            if(std::isnan(a) && std::isnan(b))
                return true;
        }
        return a == Catch::Approx(b).margin(std::numeric_limits<T>::epsilon());
    }

    //! Is complex number considered finite for math testing.
    //! Complex numbers with absolute value close to max() of underlying type are considered infinite.
    //! The reason is, CUDA/HIP implementation cannot guarantee correct treatment of such values due to
    //! implementing some math functions via calls to others. For extreme values of arguments, it could cause
    //! intermediate results to become infinite or NaN. So in this function we consider all large enough values to
    //! be effectively infinite and equivalent to one another. Thus, the tests do not concern accuracy for extreme
    //! values. However, they still check the implementation for "reasonable" values.
    template<typename T>
    constexpr bool isFinite(alpaka::math::Complex<T> const& z)
    {
        auto const absValue = abs(z);
        auto const maxAbs = static_cast<T>(0.1) * std::numeric_limits<T>::max();
        return std::isfinite(absValue) && (absValue < maxAbs);
    }

    //! Approximate comparison of complex numbers
    template<typename T>
    constexpr bool isApproxEqual(alpaka::math::Complex<T> const& a, alpaka::math::Complex<T> const& b)
    {
        // Consider all infinite values equal, @see comment at isFinite()
        if(!isFinite(a) && !isFinite(b))
            return true;
        // For the same reason use relative difference comparison with a large margin
        auto const scalingFactor = static_cast<T>(std::is_same_v<T, float> ? 1.5e4 : 1.1e6);
        auto const marginValue = scalingFactor * std::numeric_limits<T>::epsilon();
        return (a.real() == Catch::Approx(b.real()).margin(marginValue).epsilon(marginValue))
               && (a.imag() == Catch::Approx(b.imag()).margin(marginValue).epsilon(marginValue));
    }

    //! Approximate comparison of simd types
    template<typename T_Type, uint32_t T_Width>
    constexpr alpaka::SimdMask<T_Type, T_Width> isApproxEqual(
        alpaka::Simd<T_Type, T_Width> const& a,
        alpaka::Simd<T_Type, T_Width> const& b)
    {
        if constexpr(std::is_integral_v<T_Type>)
        {
            return a == b;
        }
        else
        {
            return alpaka::SimdMask<T_Type, T_Width>{
                [&a, &b](uint32_t id) -> bool
                { return a[id] == Catch::Approx(b[id]).margin(std::numeric_limits<T_Type>::epsilon()); }};
        }
    }
} // namespace alpaka::test
