/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
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
                    return ::remainder(x, y);
                }
            };
            //! The CUDA remainder float specialization.
            template<>
            struct Remainder<RemainderUniformCudaHipBuiltIn, float, float>
            {
                __device__ auto operator()(
                    RemainderUniformCudaHipBuiltIn const& remainder_ctx,
                    float const& x,
                    float const& y) -> float
                {
                    alpaka::ignore_unused(remainder_ctx);
                    return ::remainderf(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
