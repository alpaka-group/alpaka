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

#include <alpaka/math/sin/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA sin.
        class SinCudaHipBuiltIn : public concepts::Implements<ConceptMathSin, SinCudaHipBuiltIn>
        {};

        namespace traits
        {
            //#############################################################################
            //! The CUDA sin trait specialization.
            template<
                typename TArg>
            struct Sin<
                SinCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto sin(
                    SinCudaHipBuiltIn const & sin_ctx,
                    TArg const & arg)
                -> decltype(::sin(arg))
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sin(arg);
                }
            };
            //! The CUDA sin float specialization.
            template<>
            struct Sin<
                SinCudaHipBuiltIn,
                float>
            {
                __device__ static auto sin(
                    SinCudaHipBuiltIn const & sin_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sinf(arg);
                }
            };
        }
    }
}

#endif
