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
#include <alpaka/math/fmod/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL fmod.
        class FmodGenericSycl : public concepts::Implements<ConceptMathFmod, FmodGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL fmod trait specialization.
            template<typename Tx, typename Ty>
            struct Fmod<FmodGenericSycl, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
            {
                static auto fmod(FmodGenericSycl const &, Tx const & x, Ty const & y)
                {
                    return cl::sycl::fmod(x, y);
                }
            };
        }
    }
}

#endif
