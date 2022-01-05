/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Jan Stephan, Andrea Bocci
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
#    include <alpaka/math/round/RoundUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/round/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA round trait specialization.
            template<typename TArg>
            struct Round<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(RoundUniformCudaHipBuiltIn const& round_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(round_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::roundf(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::round(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };

            //! The CUDA lround trait specialization.
            template<typename TArg>
            struct Lround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(RoundUniformCudaHipBuiltIn const& lround_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(lround_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::lroundf(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::lround(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };

            //! The CUDA llround trait specialization.
            template<typename TArg>
            struct Llround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(RoundUniformCudaHipBuiltIn const& llround_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(llround_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::llroundf(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::llround(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
