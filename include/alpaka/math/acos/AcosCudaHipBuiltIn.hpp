 /* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
 
#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #include <cuda_runtime.h>
    #if !BOOST_LANG_CUDA
        #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
    #endif
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)

    #if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
        #include <cuda_runtime_api.h>
    #else
        #if BOOST_COMP_HCC
            #include <math_functions.h>
        #else
            #include <math_functions.hpp>
        #endif
    #endif
    
    #if !BOOST_LANG_HIP
        #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
    #endif
#endif

#include <alpaka/core/Unused.hpp>

#include <type_traits>

#include <alpaka/math/acos/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in acos.
        class AcosCudaHipBuiltIn : public concepts::Implements<ConceptMathAcos, AcosCudaHipBuiltIn>
        {};

        namespace traits
        {
            //#############################################################################
            //! The CUDA acos trait specialization.
            template<
                typename TArg>
            struct Acos<
                AcosCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto acos(
                    AcosCudaHipBuiltIn const & acos_ctx,
                    TArg const & arg)
                -> decltype(::acos(arg))
                {
                    alpaka::ignore_unused(acos_ctx);
                    return ::acos(arg);
                }
            };

            template<>
            struct Acos<
                AcosCudaHipBuiltIn,
                float>
            {
                __device__ static auto acos(
                    AcosCudaHipBuiltIn const & acos_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(acos_ctx);
                    return ::acosf(arg);
                }
            };
        }
    }
}

#endif
