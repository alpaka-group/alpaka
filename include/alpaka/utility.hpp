/* Copyright 2024 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/core/common.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <concepts>
#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace core
    {
        //! convert any type to a reference type
        //
        // This function is equivalent to std::declval() but can be used
        // within an alpaka accelerator kernel too.
        // This function can be used only within std::decltype().
#if ALPAKA_LANG_CUDA && ALPAKA_COMP_CLANG_CUDA || ALPAKA_COMP_HIP
        template<class T>
        ALPAKA_FN_HOST_ACC std::add_rvalue_reference_t<T> declval();
#else
        using std::declval;
#endif
    } // namespace core

    /// Returns the ceiling of a / b, as integer.
    template<std::integral T_Integral>
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto divCeil(T_Integral a, T_Integral b) -> T_Integral
    {
        return (a + b - T_Integral{1}) / b;
    }

    /// Returns the  max(a / b, 1) as integer.
    template<std::integral T_Integral>
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto divExZero(T_Integral a, T_Integral b) -> T_Integral
    {
        return std::max(a / b, T_Integral{1});
    }

    /// Computes the nth power of base, in integers.
    template<typename T_Integral, typename = std::enable_if_t<std::is_integral_v<T_Integral>>>
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto intPow(T_Integral base, T_Integral n) -> T_Integral
    {
        if(n == 0)
            return 1;
        auto r = base;
        for(T_Integral i = 1; i < n; i++)
            r *= base;
        return r;
    }

    /// Computes the floor of the nth root of value, in integers.
    template<typename T_Integral, typename = std::enable_if_t<std::is_integral_v<T_Integral>>>
    [[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto nthRootFloor(T_Integral value, T_Integral n) -> T_Integral
    {
        // adapted from: https://en.wikipedia.org/wiki/Integer_square_root
        T_Integral l = 0;
        T_Integral r = value + 1;
        while(l != r - 1)
        {
            T_Integral const m = (l + r) / 2;
            if(intPow(m, n) <= value)
                l = m;
            else
                r = m;
        }
        return l;
    }

    template<std::integral T>
    inline constexpr T firstSetBit(T value)
    {
        using UnsignedValueType = std::make_unsigned_t<T>;
        return sizeof(T) * CHAR_BIT - 1 - std::countl_zero(static_cast<UnsignedValueType>(value));
    }

    /** round to the next power of two which is equal or lower to the value
     *
     * @param value input value >0
     */
    template<std::integral T>
    inline constexpr T roundDownToPowerOfTwo(T value)
    {
        using UnsignedValueType = std::make_unsigned_t<T>;
        return static_cast<T>(std::bit_floor(static_cast<UnsignedValueType>(value)));
    }

    /** checks if T is a instance of U
     *
     * @tparam T full type specialization
     * @tparam U unspecialized template type
     *
     * @return true if T is a specialization of U
     *
     * @{
     */
    template<typename T, template<typename...> typename U>
    inline constexpr bool isSpecializationOf_v = std::false_type{};

    template<template<typename...> typename U, typename... Vs>
    inline constexpr bool isSpecializationOf_v<U<Vs...>, U> = std::true_type{};

    /** @} */

    namespace concepts
    {
        /** Validates if T is a specialization of the unspecialized template type U.
         *
         * @tparam T full type specialization
         * @tparam U unspecialized template type
         */
        template<typename T, template<typename...> typename U>
        concept SpecializationOf = isSpecializationOf_v<std::remove_cvref_t<T>, U>;
    } // namespace concepts

    /**
     * @brief Helper function calculating the integer power for the given base and exponent.
     */
    constexpr auto ipow(std::integral auto const base, std::integral auto const exponent)
        requires std::same_as<ALPAKA_TYPEOF(base), ALPAKA_TYPEOF(exponent)>
    {
        using Result = ALPAKA_TYPEOF(base);
        Result result = Result{1};
        if(exponent == Result{0})
            return result;

        result = ipow(base, exponent / Result{2});
        result *= result;

        if(exponent % Result{2})
            result *= base;

        return result;
    }
} // namespace alpaka
