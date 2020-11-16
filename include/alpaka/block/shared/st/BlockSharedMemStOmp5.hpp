/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/block/shared/st/Traits.hpp>
#    include <alpaka/block/shared/st/BlockSharedMemStMember.hpp>

#    include <type_traits>
#    include <cstdint>
#    include <omp.h>

namespace alpaka
{
    //#############################################################################
    //! The OpenMP 5 block shared memory allocator.
    class BlockSharedMemStOmp5
        : public detail::BlockSharedMemStMemberImpl<4>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStOmp5>
    {
    public:
        using BlockSharedMemStMemberImpl<4>::BlockSharedMemStMemberImpl;
    };

    namespace traits
    {
        //#############################################################################
        template<typename T, std::size_t TuniqueId>
        struct AllocVar<T, TuniqueId, BlockSharedMemStOmp5>
        {
            //-----------------------------------------------------------------------------
            static auto allocVar(BlockSharedMemStOmp5 const& smem) -> T&
            {
#    pragma omp barrier
                smem.alloc<T>();
#    pragma omp barrier
                return smem.getLatestVar<T>();
            }
        };
        //#############################################################################
        template<>
        struct FreeMem<BlockSharedMemStOmp5>
        {
            //-----------------------------------------------------------------------------
            static auto freeMem(BlockSharedMemStOmp5 const& mem) -> void
            {
                mem.free();
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
