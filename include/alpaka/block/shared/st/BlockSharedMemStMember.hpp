/* Copyright 2020 Jeffrey Kelling, Rene Widera
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
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! Static block shared memory provider using a pointer to
    //! externally allocated fixed-size memory, likely provided by
    //! BlockSharedMemDynMember.
    //! \warning This class is not thread safe!
    template<std::size_t TDataAlignBytes = core::vectorization::defaultAlignment>
    class BlockSharedMemStMember
        : public detail::BlockSharedMemStMemberImpl<TDataAlignBytes>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStMember<TDataAlignBytes>>
    {
    public:
        using detail::BlockSharedMemStMemberImpl<TDataAlignBytes>::BlockSharedMemStMemberImpl;
    };

    namespace traits
    {
        //#############################################################################
        template<typename T, std::size_t TDataAlignBytes, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStMember<TDataAlignBytes>>
        {
            //-----------------------------------------------------------------------------
            static auto declareVar(BlockSharedMemStMember<TDataAlignBytes> const& smem) -> T&
            {
                auto* data = smem.template getVarPtr<T>(TuniqueId);

                if(!data)
                {
                    smem.template alloc<T>(TuniqueId);
                    data = smem.template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
            }
        };
        //#############################################################################
        template<std::size_t TDataAlignBytes>
        struct FreeSharedVars<BlockSharedMemStMember<TDataAlignBytes>>
        {
            //-----------------------------------------------------------------------------
            static auto freeVars(BlockSharedMemStMember<TDataAlignBytes> const&) -> void
            {
                // shared memory block data will be reused
            }
        };
    } // namespace traits
} // namespace alpaka
