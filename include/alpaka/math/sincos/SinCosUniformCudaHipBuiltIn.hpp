/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Jan Stephan, Andrea Bocci
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
#    include <alpaka/math/sincos/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA sincos.
        class SinCosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosUniformCudaHipBuiltIn>
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
            //! The CUDA sincos trait specialization.
            template<typename TArg>
            struct SinCos<SinCosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                __device__ auto operator()(
                    SinCosUniformCudaHipBuiltIn const& sincos_ctx,
                    TArg const& arg,
                    TArg& result_sin,
                    TArg& result_cos) -> void
                {
                    alpaka::ignore_unused(sincos_ctx);

                    if constexpr(is_decayed_v<TArg, float>)
                        ::sincosf(arg, &result_sin, &result_cos);
                    else if constexpr(is_decayed_v<TArg, double>)
                        ::sincos(arg, &result_sin, &result_cos);
                    else
                        static_assert(!sizeof(TArg), "Unsupported data type");
                }
            };
        } // namespace traits

#    endif

    } // namespace math
} // namespace alpaka

#endif
