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
#    include <alpaka/math/remainder/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in remainder.
        class RemainderUniformCudaHipBuiltIn
            : public concepts::Implements<ConceptMathRemainder, RemainderUniformCudaHipBuiltIn>
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
            //! The CUDA remainder trait specialization.
            template<typename Tx, typename Ty>
            struct Remainder<
                RemainderUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_floating_point<Tx>::value && std::is_floating_point<Ty>::value>>
            {
                __device__ auto operator()(
                    RemainderUniformCudaHipBuiltIn const& remainder_ctx,
                    Tx const& x,
                    Ty const& y)
                {
                    alpaka::ignore_unused(remainder_ctx);

                    if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
                        return ::remainderf(x, y);
                    else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
                        return ::remainder(x, y);
                    else
                        static_assert(!sizeof(Tx), "Unsupported data type");
                }
            };
        } // namespace traits

#    endif

    } // namespace math
} // namespace alpaka

#endif
