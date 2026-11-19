/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci, René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file
 * @brief This file contains specializations of methods where we do not want to fall back to `std::*` functions.
 */

#include "alpaka/core/Unreachable.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/math/internal/ieee754.hpp"
#include "alpaka/math/internal/math.hpp"
#include "alpaka/math/internal/stlMath.hpp"

#include <cmath>
#include <type_traits>

namespace alpaka::math::internal
{
    template<typename T_A, typename T_B>
    requires(std::is_arithmetic_v<T_A> && std::is_arithmetic_v<T_B>)
    struct Min::Op<StlMath, T_A, T_B>
    {
        constexpr auto operator()(StlMath, T_A const& a, T_B const& b) const
        {
            if constexpr(std::is_integral_v<T_A> && std::is_integral_v<T_B>)
            {
                using std::min;
                return min(a, b);
            }
            else if constexpr(
                isDecayed_v<T_A, float> || isDecayed_v<T_B, float> || isDecayed_v<T_A, double>
                || isDecayed_v<T_B, double>)
            {
                using std::fmin;
                return fmin(a, b);
            }
            else
                static_assert(!sizeof(T_A), "Unsupported data type");

            ALPAKA_UNREACHABLE(std::common_type_t<T_A, T_B>{});
        }
    };

    template<typename T_A, typename T_B>
    requires(std::is_arithmetic_v<T_A> && std::is_arithmetic_v<T_B>)
    struct Max::Op<StlMath, T_A, T_B>
    {
        constexpr auto operator()(StlMath, T_A const& a, T_B const& b) const
        {
            if constexpr(std::is_integral_v<T_A> && std::is_integral_v<T_B>)
            {
                using std::max;
                return max(a, b);
            }
            else if constexpr(
                isDecayed_v<T_A, float> || isDecayed_v<T_B, float> || isDecayed_v<T_A, double>
                || isDecayed_v<T_B, double>)
            {
                using std::fmax;
                return fmax(a, b);
            }
            else
                static_assert(!sizeof(T_A), "Unsupported data type");

            ALPAKA_UNREACHABLE(std::common_type_t<T_A, T_B>{});
        }
    };

    //! Custom IEEE 754 bitwise implementation of isnan
    //! std counterpart does not work correctly for `-ffast-math` flags at CPU.
    template<std::floating_point T_Arg>
    struct Isnan::Op<StlMath, T_Arg>
    {
        constexpr auto operator()(StlMath, T_Arg const& arg) const -> bool
        {
            return ieeeIsnan(arg);
        }
    };

    //! Custom IEEE 754 bitwise implementation of isinf
    //! std counterpart does not work correctly for `-ffast-math` flags at CPU.
    template<std::floating_point T_Arg>
    struct Isinf::Op<StlMath, T_Arg>
    {
        constexpr auto operator()(StlMath, T_Arg const& arg) const -> bool
        {
            return ieeeIsinf(arg);
        }
    };

    //! Custom IEEE 754 bitwise implementation of isinf
    //! std counterpart does not work correctly for `-ffast-math` flags at CPU.
    template<std::floating_point T_Arg>
    struct Isfinite::Op<StlMath, T_Arg>
    {
        constexpr auto operator()(StlMath, T_Arg const& arg) const -> bool
        {
            return ieeeIsfinite(arg);
        }
    };

    //! Custom IEEE 754 bitwise implementation of isnan
    //! std counterpart does not work correctly for `-ffast-math` flags at CPU.
} // namespace alpaka::math::internal
