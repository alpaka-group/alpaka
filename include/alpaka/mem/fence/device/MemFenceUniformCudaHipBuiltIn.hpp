/* Copyright 2022 Jan Stephan, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/DeviceOnly.hpp>
#    include <alpaka/mem/fence/MemFenceUniformCudaHipBuiltIn.hpp>

namespace alpaka
{
    namespace traits
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
        struct MemFence<MemFenceUniformCudaHipBuiltIn, memory_scope::Device>
        {
            __device__ static auto mem_fence(MemFenceUniformCudaHipBuiltIn const&, memory_scope::Device const&)
            {
                __threadfence();
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
