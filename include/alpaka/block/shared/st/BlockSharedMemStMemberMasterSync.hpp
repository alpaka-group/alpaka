/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp>
#include <alpaka/core/AlignedAlloc.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <functional>
#include <memory>
#include <vector>


namespace alpaka
{
    template<std::size_t TDataAlignBytes = core::vectorization::defaultAlignment>
    class BlockSharedMemStMemberMasterSync
        : public detail::BlockSharedMemStMemberImpl<TDataAlignBytes>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStMemberMasterSync<TDataAlignBytes>>
    {
    public:
        BlockSharedMemStMemberMasterSync(
            uint8_t* mem,
            std::size_t capacity,
            std::function<void()> fnSync,
            std::function<bool()> fnIsMasterThread)
            : detail::BlockSharedMemStMemberImpl<TDataAlignBytes>(mem, capacity)
            , m_syncFn(fnSync)
            , m_isMasterThreadFn(fnIsMasterThread)
        {
        }

        std::function<void()> m_syncFn;
        std::function<bool()> m_isMasterThreadFn;
    };

    namespace traits
    {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
        //#############################################################################
        template<typename T, std::size_t TDataAlignBytes, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStMemberMasterSync<TDataAlignBytes>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto declareVar(
                BlockSharedMemStMemberMasterSync<TDataAlignBytes> const& blockSharedMemSt) -> T&
            {
                auto* data = blockSharedMemSt.template getVarPtr<T>(TuniqueId);

                if(!data)
                {
                    // Assure that all threads have executed the return of the last allocBlockSharedArr function (if
                    // there was one before).
                    blockSharedMemSt.m_syncFn();
                    if(blockSharedMemSt.m_isMasterThreadFn())
                    {
                        blockSharedMemSt.template alloc<T>(TuniqueId);
                    }

                    blockSharedMemSt.m_syncFn();
                    // lookup for the data chunk allocated by the master thread
                    data = blockSharedMemSt.template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
            }
        };
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        //#############################################################################
        template<std::size_t TDataAlignBytes>
        struct FreeSharedVars<BlockSharedMemStMemberMasterSync<TDataAlignBytes>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto freeVars(BlockSharedMemStMemberMasterSync<TDataAlignBytes> const&) -> void
            {
                // shared memory block data will be reused
            }
        };
    } // namespace traits
} // namespace alpaka
