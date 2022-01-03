/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Valentin Gehrke, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/CudaHipMath.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/math/rsqrt/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA rsqrt.
        class RsqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA rsqrt trait specialization.
            template<typename TArg>
            struct Rsqrt<RsqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                __device__ auto operator()(RsqrtUniformCudaHipBuiltIn const& rsqrt_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(rsqrt_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::rsqrtf(arg);
                    else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                        return ::rsqrt(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
