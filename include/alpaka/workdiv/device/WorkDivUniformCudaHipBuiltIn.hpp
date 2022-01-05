/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <alpaka/core/DeviceOnly.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/idx/Traits.hpp>
#    include <alpaka/vec/Vec.hpp>
#    include <alpaka/workdiv/WorkDivUniformCudaHipBuiltIn.hpp>

namespace alpaka
{
    namespace traits
    {
        //! The GPU CUDA/HIP accelerator work division dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The GPU CUDA/HIP accelerator work division idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The GPU CUDA/HIP accelerator work division grid block extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The number of blocks in each dimension of the grid.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& workDiv)
                -> Vec<TDim, TIdx>
            {
                alpaka::ignore_unused(workDiv);
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return castVec<TIdx>(extent::getExtentVecEnd<TDim>(gridDim));
#    else
                return extent::getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipGridDim_z),
                    static_cast<TIdx>(hipGridDim_y),
                    static_cast<TIdx>(hipGridDim_x)));
#    endif
            }
        };

        //! The GPU CUDA/HIP accelerator work division block thread extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The number of threads in each dimension of a block.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& workDiv)
                -> Vec<TDim, TIdx>
            {
                alpaka::ignore_unused(workDiv);
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return castVec<TIdx>(extent::getExtentVecEnd<TDim>(blockDim));
#    else
                return extent::getExtentVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipBlockDim_z),
                    static_cast<TIdx>(hipBlockDim_y),
                    static_cast<TIdx>(hipBlockDim_x)));
#    endif
            }
        };

        //! The GPU CUDA/HIP accelerator work division thread element extent trait specialization.
        template<typename TDim, typename TIdx>
        struct GetWorkDiv<WorkDivUniformCudaHipBuiltIn<TDim, TIdx>, origin::Thread, unit::Elems>
        {
            //! \return The number of blocks in each dimension of the grid.
            __device__ static auto getWorkDiv(WorkDivUniformCudaHipBuiltIn<TDim, TIdx> const& workDiv)
                -> Vec<TDim, TIdx>
            {
                return workDiv.m_threadElemExtent;
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
