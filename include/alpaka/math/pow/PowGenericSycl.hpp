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
#include <alpaka/math/pow/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL pow.
        class PowGenericSycl : public concepts::Implements<ConceptMathPow, PowGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL pow trait specialization.
            template<typename TBase, typename TExp>
            struct Pow<PowGenericSycl, TBase, TExp, std::enable_if_t<std::is_arithmetic_v<TBase> && std::is_arithmetic_v<TExp>>>
            {
                static auto pow(PowGenericSycl const &, TBase const & base, TExp const & exp)
                {
                    return cl::sycl::pow(base, exp);
                }
            };
        }
    }
}

#endif
