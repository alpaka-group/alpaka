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
#include <alpaka/math/sin/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL library sin.
        class SinGenericSycl : public concepts::Implements<ConceptMathSin, SinGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL library sin trait specialization.
            template<typename TArg>
            struct Sin<SinGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                static auto sin(SinGenericSycl const &, TArg const & arg)
                {
                    return cl::sycl::sin(arg);
                }
            };
        }
    }
}

#endif
