/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace alpaka
{
    /** Marks a functor which supports SimdPtr as arguments
     *
     * Wrapping a functor or lambda with this class to signal support for SimdPtr.
     * A stencil functor can be used to write stencil operations within a transform call.
     */
    template<typename T_Func>
    struct StencilFunc : T_Func
    {
        using Functor = T_Func;
        using T_Func::T_Func;

        /* StencilFunc is a stateless wrapper type
         * NOLINT reason: Constraining the wrapper's forwarding constructor to exclude its own type does not merely
         * prevent accidental constructor shadowing. It can also change the constructor-selection behavior of the
         * wrapped type.
         */
        // NOLINTBEGIN(bugprone-forwarding-reference-overload)
        constexpr StencilFunc(auto&& func) : T_Func{ALPAKA_FORWARD(func)}
        {
        }

        // NOLINTEND(bugprone-forwarding-reference-overload)
    };

    template<typename T_Func>
    ALPAKA_FN_DG StencilFunc(T_Func&&) -> StencilFunc<T_Func>;

    /** Marks a functor that can only be executed with scalar types and not SIMD packages.
     *
     * The functor will be executed element wise for SIMD packages due to methods used which prevent using SIMD
     * packages directly.
     */
    template<typename T_Func>
    struct ScalarFunc : T_Func
    {
        using Functor = T_Func;
        using T_Func::T_Func;

        /* ScalarFunc is a stateless wrapper type
         * NOLINT reason: Constraining the wrapper's forwarding constructor to exclude its own type does not merely
         * prevent accidental constructor shadowing. It can also change the constructor-selection behavior of the
         * wrapped type.
         */
        // NOLINTBEGIN(bugprone-forwarding-reference-overload)
        constexpr ScalarFunc(auto&& func) : T_Func{ALPAKA_FORWARD(func)}
        {
        }

        // NOLINTEND(bugprone-forwarding-reference-overload)
    };

    template<typename T_Func>
    ALPAKA_FN_DG ScalarFunc(T_Func&&) -> ScalarFunc<T_Func>;

    /** Execute the functor with or without an accelerator as first argument
     *
     * The functor is not allowed to have both possible signatures.
     *
     * @{
     */
    template<typename T_Acc, typename T_Functor, typename... T_Args>
    requires std::invocable<T_Functor, T_Acc, T_Args...>
    inline constexpr auto callFunctor(T_Acc const& acc, T_Functor&& functor, T_Args&&... args)
    {
        return functor(acc, std::forward<T_Args>(args)...);
    }

    template<typename T_Acc, typename T_Functor, typename... T_Args>
    requires std::invocable<T_Functor, T_Args...>
    inline constexpr auto callFunctor(T_Acc const&, T_Functor&& functor, T_Args&&... args)
    {
        return functor(std::forward<T_Args>(args)...);
    }

    /** @} */
} // namespace alpaka
