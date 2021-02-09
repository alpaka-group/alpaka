/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/math/max/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL library max.
        class MaxGenericSycl : public concepts::Implements<ConceptMathMax, MaxGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL integral max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<MaxGenericSycl, Tx, Ty, std::enable_if_t<std::is_integral_v<Tx> && std::is_integral_v<Ty>>>
            {
                static auto max(MaxGenericSycl const &, Tx const & x, Ty const & y)
                {
                    return cl::sycl::max(x, y);
                }
            };
            //#############################################################################
            //! The SYCL mixed integral floating point max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<MaxGenericSycl, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>
                                                                && !(std::is_integral_v<Tx> && std::is_integral_v<Ty>)>>
            {
                static auto max(MaxGenericSycl const &, Tx const & x, Ty const & y)
                {
                    return cl::sycl::fmax(x, y);
                }
            };
        }
    }
}

#endif
