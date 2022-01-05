/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, René Widera, Jan Stephan, Andrea Bocci
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
#    include <alpaka/math/trunc/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA trunc.
        class TruncUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTrunc, TruncUniformCudaHipBuiltIn>
        {
        };

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA                                                       \
        || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP

        namespace traits
        {
            //! The CUDA trunc trait specialization.
            template<typename TArg>
            struct Trunc<TruncUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(TruncUniformCudaHipBuiltIn const& trunc_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(trunc_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        return ::truncf(arg);
                    else if constexpr(is_decayed_v<TArg, double>)
                        return ::trunc(arg);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };
        } // namespace traits

#    endif

    } // namespace math
} // namespace alpaka

#endif
