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

#include <alpaka/meta/StaticFor.hpp>

namespace alpaka
{
    namespace meta
    {
        /** @brief An alternative to \a StaticFor for functions with no parameters
         *
         *  Calls the function repeatedly, but doesn't pass the counter into the loop.
         */
        template<unsigned N>
        struct RepeatN
        {
            template<typename Fn>
            constexpr void operator()(Fn&& fn) const
            {
                StaticFor<0, N>()([&](unsigned i) { fn(); });
            }
        };
    } // namespace meta
} // namespace alpaka
