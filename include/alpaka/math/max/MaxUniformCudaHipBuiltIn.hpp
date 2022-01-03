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
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/math/max/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in max.
        class MaxUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<
                MaxUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
            {
                __device__ auto operator()(MaxUniformCudaHipBuiltIn const& max_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(max_ctx);

                    if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
                        return ::max(x, y);
                    else if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                        return ::fmaxf(x, y);
                    else if constexpr(
                        is_decayed_v<
                            Tx,
                            double> || is_decayed_v<Ty, double> || (is_decayed_v<Tx, float> && std::is_integral_v<Ty>)
                        || (std::is_integral_v<Tx> && is_decayed_v<Ty, float>) )
                        return ::fmax(x, y);
                    else
                        static_assert(!sizeof(Tx), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
