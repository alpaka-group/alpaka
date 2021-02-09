/* Copyright 2020 Jan Stephan
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
#include <alpaka/math/atan/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL atan.
        class AtanGenericSycl : public concepts::Implements<ConceptMathAtan, AtanGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL atan trait specialization.
            template<typename TArg>
            struct Atan<AtanGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                static auto atan(AtanGenericSycl const&, TArg const & arg)
                {
                    return cl::sycl::atan(arg);
                }
            };
        }
    }
}

#endif
