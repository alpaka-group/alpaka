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
#    include <alpaka/math/erf/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in erf.
        class ErfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathErf, ErfUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA erf trait specialization.
            template<typename TArg>
            struct Erf<ErfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                __device__ auto operator()(ErfUniformCudaHipBuiltIn const& /* erf_ctx */, TArg const& arg)
                {
                    if constexpr(is_decayed_v<TArg, float>)
                        return ::erff(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::erf(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");

                    ALPAKA_UNREACHABLE(TArg{});
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
