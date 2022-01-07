/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Jan Stephan
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
#    include <alpaka/core/Unreachable.hpp>
#    include <alpaka/math/cos/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in cos.
        class CosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCos, CosUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA cos trait specialization.
            template<typename TArg>
            struct Cos<CosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                __device__ auto operator()(CosUniformCudaHipBuiltIn const& /* cos_ctx */, TArg const& arg)
                {
                    if constexpr(is_decayed_v<TArg, float>)
                        return ::cosf(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::cos(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");

                    ALPAKA_UNREACHABLE(TArg{});
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
