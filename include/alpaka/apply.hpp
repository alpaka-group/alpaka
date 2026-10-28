/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <utility>

namespace alpaka
{
    namespace detail
    {
        template<typename T_Func, typename T_TupleLike, std::size_t... T_idx>
        ALPAKA_FN_INLINE constexpr decltype(auto) applyImpl(
            T_Func&& func,
            T_TupleLike&& tuple,
            std::index_sequence<T_idx...>)
        {
            using std::get;
            // like std::apply: each get<T_idx> only moves its own element out of the forwarded tuple
            // NOLINTNEXTLINE(bugprone-use-after-move)
            return std::forward<T_Func>(func)(get<T_idx>(std::forward<T_TupleLike>(tuple))...);
        }
    } // namespace detail

    /** Applies a function to the elements of a tuple-like object.
     *
     * This function forwards the function and the tuple-like object, and uses an index sequence to unpack the tuple.
     *
     * @param func The function to apply.
     * @param tuple The tuple-like object containing the arguments for the function.
     * @return The result of applying the function to the elements of the tuple-like object.
     */
    template<typename T_Func, typename T_TupleLike>
    ALPAKA_FN_INLINE constexpr decltype(auto) apply(T_Func&& func, T_TupleLike&& tuple)
    {
        /** @attention Do not use std::tuple_size_v here because it results in compile issues with gcc11.4 */
        return detail::applyImpl(
            std::forward<T_Func>(func),
            std::forward<T_TupleLike>(tuple),
            std::make_index_sequence<std::tuple_size<std::decay_t<T_TupleLike>>::value>{});
    }
} // namespace alpaka
