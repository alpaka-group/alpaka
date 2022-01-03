/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Decay.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/math/min/Traits.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The standard library min.
        class MinStdLib : public concepts::Implements<ConceptMathMin, MinStdLib>
        {
        };

        namespace traits
        {
            //! The standard library min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<MinStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
            {
                ALPAKA_FN_HOST auto operator()(MinStdLib const& min_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(min_ctx);

                    using std::fmin;
                    using std::min;

                    if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                        return min(x, y);
                    else if constexpr(
                        is_decayed_v<
                            Tx,
                            float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                        return fmin(x, y);
                    else
                        static_assert(!sizeof(Tx), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka
