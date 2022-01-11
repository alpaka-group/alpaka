/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Decay.hpp>
#include <alpaka/math/max/Traits.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The standard library max.
        class MaxStdLib : public concepts::Implements<ConceptMathMax, MaxStdLib>
        {
        };

        namespace traits
        {
            //! The standard library max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<MaxStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
            {
                ALPAKA_FN_HOST auto operator()(MaxStdLib const& /* max_ctx */, Tx const& x, Ty const& y)
                {
                    using std::fmax;
                    using std::max;

                    if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                        return max(x, y);
                    else if constexpr(
                        is_decayed_v<
                            Tx,
                            float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                        return fmax(x, y);
                    else
                        static_assert(!sizeof(Tx), "Unsupported data type");

                    ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka
