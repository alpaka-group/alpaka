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
#include <alpaka/math/abs/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL library abs.
        class AbsGenericSycl : public concepts::Implements<ConceptMathAbs, AbsGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL abs trait specialization.
            template<typename TArg>
            struct Abs<AbsGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg> && std::is_signed_v<TArg>>>
            {
                static auto abs(AbsGenericSycl const&, TArg const & arg)
                {
                    return cl::sycl::fabs(arg);
                }
            };
        }
    }
}

#endif
