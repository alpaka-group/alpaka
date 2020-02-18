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

#include <alpaka/math/erf/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in erf.
        class ErfCudaHipBuiltIn : public concepts::Implements<ConceptMathErf, ErfCudaHipBuiltIn>
        {};
        namespace traits
        {
            //#############################################################################
            //! The CUDA erf trait specialization.
            template<
                typename TArg>
            struct Erf<
                ErfCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto erf(
                    ErfCudaHipBuiltIn const & erf_ctx,
                    TArg const & arg)
                -> decltype(::erf(arg))
                {
                    alpaka::ignore_unused(erf_ctx);
                    return ::erf(arg);
                }
            };

            template<>
            struct Erf<
                ErfCudaHipBuiltIn,
                float>
            {
                __device__ static auto erf(
                    ErfCudaHipBuiltIn const & erf_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(erf_ctx);
                    return ::erff(arg);
                }
            };
        }
    }
}

#endif
