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

#include <alpaka/math/round/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA round.
        class RoundCudaHipBuiltIn : public concepts::Implements<ConceptMathRound, RoundCudaHipBuiltIn>
        {};

        namespace traits
        {
            //#############################################################################
            //! The CUDA round trait specialization.
            template<
                typename TArg>
            struct Round<
                RoundCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto round(
                    RoundCudaHipBuiltIn const & round_ctx,
                    TArg const & arg)
                -> decltype(::round(arg))
                {
                    alpaka::ignore_unused(round_ctx);
                    return ::round(arg);
                }
            };
            //#############################################################################
            //! The CUDA lround trait specialization.
            template<
                typename TArg>
            struct Lround<
                RoundCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto lround(
                    RoundCudaHipBuiltIn const & lround_ctx,
                    TArg const & arg)
                -> long int
                {
                    alpaka::ignore_unused(lround_ctx);
                    return ::lround(arg);
                }
            };
            //#############################################################################
            //! The CUDA llround trait specialization.
            template<
                typename TArg>
            struct Llround<
                RoundCudaHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto llround(
                    RoundCudaHipBuiltIn const & llround_ctx,
                    TArg const & arg)
                -> long int
                {
                    alpaka::ignore_unused(llround_ctx);
                    return ::llround(arg);
                }
            };
            //! The CUDA round float specialization.
            template<>
            struct Round<
                RoundCudaHipBuiltIn,
                float>
            {
                __device__ static auto round(
                    RoundCudaHipBuiltIn const & round_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(round_ctx);
                    return ::roundf(arg);
                }
            };
        }
    }
}

#endif
