/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Valentin Gehrke, Jan Stephan, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/CudaHipMath.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/math/sqrt/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA sqrt.
        class SqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSqrt, SqrtUniformCudaHipBuiltIn>
        {
        };

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA                                                       \
        || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP

        namespace traits
        {
            //! The CUDA sqrt trait specialization.
            template<typename TArg>
            struct Sqrt<SqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                __device__ auto operator()(SqrtUniformCudaHipBuiltIn const& sqrt_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(sqrt_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::sqrtf(arg);
                    else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                        return ::sqrt(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };
        } // namespace traits

#    endif

    } // namespace math
} // namespace alpaka

#endif
