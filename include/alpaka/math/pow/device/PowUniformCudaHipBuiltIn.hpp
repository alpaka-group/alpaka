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
#    include <alpaka/math/pow/PowUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/pow/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA pow trait specialization.
            template<typename TBase, typename TExp>
            struct Pow<
                PowUniformCudaHipBuiltIn,
                TBase,
                TExp,
                std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
            {
                __device__ auto operator()(PowUniformCudaHipBuiltIn const& pow_ctx, TBase const& base, TExp const& exp)
                {
                    alpaka::ignore_unused(pow_ctx);

                    if constexpr(is_decayed_v<TBase, float> && is_decayed_v<TExp, float>)
                        return ::powf(base, exp);
                    else if constexpr(is_decayed_v<TBase, double> || is_decayed_v<TExp, double>)
                        return ::pow(base, exp);
                    else
                        static_assert(!sizeof(TBase), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
