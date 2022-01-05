/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Jeffrey Kelling, Andrea Bocci
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
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/math/isinf/IsinfUniformCudaHipBuiltIn.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA isinf trait specialization.
            template<typename TArg>
            struct Isinf<IsinfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(IsinfUniformCudaHipBuiltIn const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    return ::isinf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
