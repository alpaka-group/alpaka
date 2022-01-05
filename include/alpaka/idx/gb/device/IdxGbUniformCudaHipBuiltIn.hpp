/* Copyright 2022 Axel Huebl, Benjamin Worpitz, René Widera, Matthias Werner, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/core/DeviceOnly.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/idx/gb/IdxGbUniformCudaHipBuiltIn.hpp>
#    include <alpaka/vec/Vec.hpp>

// Backend specific includes.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

namespace alpaka
{
    namespace traits
    {
        //! The GPU CUDA/HIP accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The GPU CUDA/HIP accelerator grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            __device__ static auto getIdx(gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx> const& idx, TWorkDiv const&)
                -> Vec<TDim, TIdx>
            {
                alpaka::ignore_unused(idx);
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return castVec<TIdx>(getOffsetVecEnd<TDim>(blockIdx));
#    else
                return getOffsetVecEnd<TDim>(Vec<std::integral_constant<typename TDim::value_type, 3>, TIdx>(
                    static_cast<TIdx>(hipBlockIdx_z),
                    static_cast<TIdx>(hipBlockIdx_y),
                    static_cast<TIdx>(hipBlockIdx_x)));
#    endif
            }
        };

        //! The GPU CUDA/HIP accelerator grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbUniformCudaHipBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#endif
