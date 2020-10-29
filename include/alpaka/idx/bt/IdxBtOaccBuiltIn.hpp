/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

#include <alpaka/idx/Traits.hpp>
#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/core/Concepts.hpp>

namespace alpaka
{
    namespace bt
    {
        //#############################################################################
        //! The CUDA accelerator ND index provider.
        template<
            typename TDim,
            typename TIdx>
        class IdxBtOaccBuiltIn : public concepts::Implements<ConceptIdxBt, IdxBtOaccBuiltIn<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            IdxBtOaccBuiltIn(TIdx blockThreadIdx) : m_blockThreadIdx(blockThreadIdx) {};
            //-----------------------------------------------------------------------------
            IdxBtOaccBuiltIn(IdxBtOaccBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            IdxBtOaccBuiltIn(IdxBtOaccBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxBtOaccBuiltIn const & ) -> IdxBtOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxBtOaccBuiltIn &&) -> IdxBtOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~IdxBtOaccBuiltIn() = default;

            const TIdx m_blockThreadIdx;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The OpenACC accelerator index dimension get trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct DimType<
            bt::IdxBtOaccBuiltIn<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The OpenACC accelerator block thread index get trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetIdx<
            bt::IdxBtOaccBuiltIn<TDim, TIdx>,
            origin::Block,
            unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the block.
            template<
                typename TWorkDiv>
            static auto getIdx(
                bt::IdxBtOaccBuiltIn<TDim, TIdx> const &idx,
                TWorkDiv const & workDiv)
            -> Vec<TDim, TIdx>
            {
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(idx.m_blockThreadIdx),
                    getWorkDiv<Block, Threads>(workDiv));
            }
        };

        template<
            typename TIdx>
        struct GetIdx<
            bt::IdxBtOaccBuiltIn<DimInt<1u>, TIdx>,
            origin::Block,
            unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the block.
            template<
                typename TWorkDiv>
            static auto getIdx(
                bt::IdxBtOaccBuiltIn<DimInt<1u>, TIdx> const & idx,
                TWorkDiv const &)
            -> Vec<DimInt<1u>, TIdx>
            {
                return idx.m_blockThreadIdx;
            }
        };

        //#############################################################################
        //! The OpenACC accelerator block thread index idx type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct IdxType<
            bt::IdxBtOaccBuiltIn<TDim, TIdx>>
        {
            using type = TIdx;
        };
    }
}

#endif
