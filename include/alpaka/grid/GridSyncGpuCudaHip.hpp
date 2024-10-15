/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/grid//Traits.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <cooperative_groups.h>
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        include <hip/hip_cooperative_groups.h>
#    endif


namespace alpaka
{
    //! The GPU CUDA/HIP grid synchronization.
    class GridSyncCudaHipBuiltIn : public concepts::Implements<ConceptGridSync, GridSyncCudaHipBuiltIn>
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
        struct SyncGridThreads<GridSyncCudaHipBuiltIn>
        {
            __device__ static auto syncGridThreads(GridSyncCudaHipBuiltIn const& /*gridSync*/) -> void
            {
                cooperative_groups::this_grid().sync();
            }
        };

    } // namespace trait

#    endif

} // namespace alpaka

#endif
