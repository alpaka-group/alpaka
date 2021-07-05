/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <functional>
#include <numeric>
#include <type_traits>

namespace alpaka
{
    namespace meta
    {
        // --- StaticFor --------------------------------------------

        /** Recursive helper for implementing static for loops.
         *
         *  @tparam First Low iteration bound
         *  @tparam Last High iteration bound
         */
        template<unsigned First, unsigned Last>
        struct StaticFor
        {
            template<typename Fn>
            constexpr void operator()(Fn&& fn) const
            {
                static_assert(First < Last, "Invalid static loop bounds");
                fn(First);
                StaticFor<First + 1, Last>()(fn);
            }
        };

        /// End of recursion for \a StaticFor
        template<unsigned N>
        struct StaticFor<N, N>
        {
            template<typename Fn>
            constexpr void operator()(Fn&& fn) const
            {
            }
        };
    } // namespace meta
} // namespace alpaka
