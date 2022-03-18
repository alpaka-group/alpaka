/* Copyright 2022 Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber
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
#    include <alpaka/mem/fence/Traits.hpp>

namespace alpaka
{
    //! The GPU CUDA/HIP memory fence.
    class MemFenceUniformCudaHipBuiltIn : public concepts::Implements<ConceptMemFence, MemFenceUniformCudaHipBuiltIn>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        template<>
        struct MemFence<MemFenceUniformCudaHipBuiltIn, memory_scope::Block>
        {
            __device__ static auto mem_fence(MemFenceUniformCudaHipBuiltIn const&, memory_scope::Block const&)
            {
                __threadfence_block();
            }
        };

        template<>
        struct MemFence<MemFenceUniformCudaHipBuiltIn, memory_scope::Grid>
        {
            __device__ static auto mem_fence(MemFenceUniformCudaHipBuiltIn const&, memory_scope::Grid const&)
            {
                // CUDA and HIP do not have a per-grid memory fence, so a device-level fence is used
                __threadfence();
            }
        };

        template<>
        struct MemFence<MemFenceUniformCudaHipBuiltIn, memory_scope::Device>
        {
            __device__ static auto mem_fence(MemFenceUniformCudaHipBuiltIn const&, memory_scope::Device const&)
            {
                __threadfence();
            }
        };
    } // namespace trait

#    endif

} // namespace alpaka

#endif
