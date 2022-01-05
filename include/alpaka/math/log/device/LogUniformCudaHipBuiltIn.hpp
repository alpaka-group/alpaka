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
#    include <alpaka/math/log/LogUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/log/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA log trait specialization.
            template<typename TArg>
            struct Log<LogUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                __device__ auto operator()(LogUniformCudaHipBuiltIn const& log_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(log_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::logf(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::log(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
