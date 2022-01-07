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
#    include <alpaka/math/atan2/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in atan2.
        class Atan2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA atan2 trait specialization.
            template<typename Ty, typename Tx>
            struct Atan2<
                Atan2UniformCudaHipBuiltIn,
                Ty,
                Tx,
                std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
            {
                __device__ auto operator()(Atan2UniformCudaHipBuiltIn const& /* atan2_ctx */, Ty const& y, Tx const& x)
                {
                    if constexpr(is_decayed_v<Ty, float> && is_decayed_v<Tx, float>)
                        return ::atan2f(y, x);
                    else if constexpr(is_decayed_v<Ty, double> || is_decayed_v<Tx, double>)
                        return ::atan2(y, x);
                    else
                        static_assert(!sizeof(Ty), "Unsupported data type");

                    ALPAKA_UNREACHABLE(Ty{});
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
