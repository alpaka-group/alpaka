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

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/CudaHipMath.hpp>
#    include <alpaka/core/Decay.hpp>
#    include <alpaka/core/Unused.hpp>
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

#    if !defined(ALPAKA_HOST_API)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

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
                __device__ auto operator()(Atan2UniformCudaHipBuiltIn const& atan2_ctx, Ty const& y, Tx const& x)
                {
                    alpaka::ignore_unused(atan2_ctx);

                    if constexpr(is_decayed_v<Ty, float> && is_decayed_v<Tx, float>)
                        return ::atan2f(y, x);
                    else if constexpr(is_decayed_v<Ty, double> || is_decayed_v<Tx, double>)
                        return ::atan2(y, x);
                    else
                        static_assert(!sizeof(Ty), "Unsupported data type");
                }
            };
        } // namespace traits

#    endif

    } // namespace math
} // namespace alpaka

#endif
