/* Copyright 2022 Benjamin Worpitz, Erik Zenker, René Widera, Matthias Werner, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/block/shared/st/BlockSharedMemStUniformCudaHipBuiltIn.hpp>
#    include <alpaka/core/DeviceOnly.hpp>

#    include <cstdint>
#    include <type_traits>

namespace alpaka
{
    namespace traits
    {
        template<typename T, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStUniformCudaHipBuiltIn>
        {
            __device__ static auto declareVar(BlockSharedMemStUniformCudaHipBuiltIn const&) -> T&
            {
                __shared__ uint8_t shMem alignas(alignof(T))[sizeof(T)];
                return *(reinterpret_cast<T*>(shMem));
            }
        };
        template<>
        struct FreeSharedVars<BlockSharedMemStUniformCudaHipBuiltIn>
        {
            __device__ static auto freeVars(BlockSharedMemStUniformCudaHipBuiltIn const&) -> void
            {
                // Nothing to do. CUDA/HIP block shared memory is automatically freed when all threads left the block.
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
