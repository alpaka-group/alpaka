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
#    include <alpaka/math/fmod/FmodUniformCudaHipBuiltIn.hpp>
#    include <alpaka/math/fmod/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA fmod trait specialization.
            template<typename Tx, typename Ty>
            struct Fmod<
                FmodUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
            {
                __device__ auto operator()(FmodUniformCudaHipBuiltIn const& fmod_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(fmod_ctx);

                    if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                        return ::fmodf(x, y);
                    else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                        return ::fmod(x, y);
                    else
                        static_assert(!sizeof(Tx), "Unsupported data type");
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
