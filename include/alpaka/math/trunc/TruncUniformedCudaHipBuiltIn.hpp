/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, René Widera
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
        #if BOOST_COMP_HCC || BOOST_COMP_HIP
            #include <math_functions.h>
        #else
            #include <math_functions.hpp>
        #endif
    #endif
    
    #if !BOOST_LANG_HIP
        #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
    #endif
#endif

#include <alpaka/math/trunc/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA trunc.
        class TruncUniformedCudaHipBuiltIn : public concepts::Implements<ConceptMathTrunc, TruncUniformedCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA trunc trait specialization.
            template<
                typename TArg>
            struct Trunc<
                TruncUniformedCudaHipBuiltIn,
                TArg,
                std::enable_if_t<
                    std::is_floating_point<TArg>::value>>
            {
                __device__ static auto trunc(
                    TruncUniformedCudaHipBuiltIn const & trunc_ctx,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(trunc_ctx);
                    return ::trunc(arg);
                }
            };
            //! The CUDA trunc float specialization.
            template<>
            struct Trunc<
                TruncUniformedCudaHipBuiltIn,
                float>
            {
                __device__ static auto trunc(
                    TruncUniformedCudaHipBuiltIn const & trunc_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(trunc_ctx);
                    return ::truncf(arg);
                }
            };
        }
    }
}

#endif
