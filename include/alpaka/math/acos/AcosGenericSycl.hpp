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
#include <alpaka/math/acos/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL acos.
        class AcosGenericSycl : public concepts::Implements<ConceptMathAcos, AcosGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL acos trait specialization.
            template<typename TArg>
            struct Acos<AcosGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                static auto acos(AcosGenericSycl const&, TArg const & arg)
                {
                    return cl::sycl::acos(arg);
                }
            };
        }
    }
}

#endif
