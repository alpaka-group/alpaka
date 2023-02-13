/* Copyright 2022 Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if(defined(ALPAKA_ACC_ANY_BT_OMP5_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED))                          \
    && _OPENMP >= 201811 // omp allocate requires OpenMP 5.0

#    include <alpaka/block/shared/st/Traits.hpp>

#    include <omp.h>

namespace alpaka
{
    //! The OpenMP 5.0 block static shared memory allocator.
    class BlockSharedMemStOmp5BuiltIn : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStOmp5BuiltIn>
    {
    };

    namespace trait
    {
        template<typename T, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStOmp5BuiltIn>
        {
            static auto declareVar(BlockSharedMemStOmp5BuiltIn const&) -> T&
            {
                static T shMem;
                // We disable icpx's warning here, because we are unsure whether the compiler is correct.
                // See discussion: https://github.com/alpaka-group/alpaka/pull/1700#issuecomment-1109745006
#    ifdef __INTEL_LLVM_COMPILER
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wopenmp-allocate"
#    endif
#    pragma omp allocate(shMem) allocator(omp_pteam_mem_alloc)
#    ifdef __INTEL_LLVM_COMPILER
#        pragma clang diagnostic pop
#    endif
                return shMem;
            }
        };
        template<>
        struct FreeSharedVars<BlockSharedMemStOmp5BuiltIn>
        {
            static auto freeVars(BlockSharedMemStOmp5BuiltIn const&) -> void
            {
                // Nothing to do. OpenMP runtime frees block shared memory at the end of the parallel region.
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
