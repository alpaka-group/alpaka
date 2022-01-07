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
#    include <alpaka/math/asin/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in asin.
        class AsinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsin, AsinUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA asin trait specialization.
            template<typename TArg>
            struct Asin<AsinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                __device__ auto operator()(AsinUniformCudaHipBuiltIn const& /* asin_ctx */, TArg const& arg)
                {
                    if constexpr(is_decayed_v<TArg, float>)
                        return ::asinf(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::asin(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");

                    ALPAKA_UNREACHABLE(TArg{});
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
