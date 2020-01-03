/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/tan/Traits.hpp>

#include <cuda_runtime.h>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA tan.
        class TanCudaBuiltIn
        {
        public:
            using TanBase = TanCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA tan trait specialization.
            template<
                typename TArg>
            struct Tan<
                TanCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto tan(
                    TanCudaBuiltIn const & tan_ctx,
                    TArg const & arg)
                -> decltype(::tan(arg))
                {
                    alpaka::ignore_unused(tan_ctx);
                    return ::tan(arg);
                }
            };
            //! The CUDA tan float specialization.
            template<>
            struct Tan<
                TanCudaBuiltIn,
                float>
            {
                __device__ static auto tan(
                    TanCudaBuiltIn const & tan_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(tan_ctx);
                    return ::tanf(arg);
                }
            };
        }
    }
}

#endif
