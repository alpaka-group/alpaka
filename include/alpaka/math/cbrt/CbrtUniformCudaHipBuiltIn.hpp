/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
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
#    include <alpaka/math/cbrt/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in cbrt.
        class CbrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCbrt, CbrtUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA cbrt trait specialization.
            template<typename TArg>
            struct Cbrt<CbrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                __device__ auto operator()(CbrtUniformCudaHipBuiltIn const& cbrt_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(cbrt_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::cbrtf(arg);
                    else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
                        return ::cbrt(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
