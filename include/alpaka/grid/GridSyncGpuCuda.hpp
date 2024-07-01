/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/grid//Traits.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <cooperative_groups.h>

namespace alpaka
{
    //! The GPU CUDA grid synchronization.
    class GridSyncCudaBuiltIn
        : public concepts::Implements<ConceptGridSync, GridSyncCudaBuiltIn>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

    namespace trait
    {
        template<>
        struct SyncGridThreads<GridSyncCudaBuiltIn>
        {
            __device__ static auto syncGridThreads(GridSyncCudaBuiltIn const& /*gridSync*/) -> void
            {
                cooperative_groups::this_grid().sync();
            }
        };

    } // namespace trait

#    endif

} // namespace alpaka

#endif
