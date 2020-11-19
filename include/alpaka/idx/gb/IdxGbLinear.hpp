/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/Traits.hpp>

namespace alpaka
{
    namespace gb
    {
        //#############################################################################
        //! General ND index provider based on a linear index.
        template<typename TDim, typename TIdx>
        class IdxGbLinear : public concepts::Implements<ConceptIdxGb, IdxGbLinear<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            IdxGbLinear(const TIdx& teamOffset = static_cast<TIdx>(0u)) : m_gridBlockIdx(teamOffset)
            {
            }
            //-----------------------------------------------------------------------------
            IdxGbLinear(IdxGbLinear const&) = delete;
            //-----------------------------------------------------------------------------
            IdxGbLinear(IdxGbLinear&&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbLinear const&) -> IdxGbLinear& = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbLinear&&) -> IdxGbLinear& = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~IdxGbLinear() = default;

            TIdx const m_gridBlockIdx;
        };
    } // namespace gb

    namespace traits
    {
        //#############################################################################
        //! The IdxGbLinear index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbLinear<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The IdxGbLinear grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbLinear<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            static auto getIdx(gb::IdxGbLinear<TDim, TIdx> const& idx, TWorkDiv const& workDiv) -> Vec<TDim, TIdx>
            {
                // \TODO: Would it be faster to precompute the index and cache it inside an array?
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(idx.m_gridBlockIdx),
                    getWorkDiv<Grid, Blocks>(workDiv));
            }
        };

        template<typename TIdx>
        struct GetIdx<gb::IdxGbLinear<DimInt<1u>, TIdx>, origin::Grid, unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            static auto getIdx(gb::IdxGbLinear<DimInt<1u>, TIdx> const& idx, TWorkDiv const&) -> Vec<DimInt<1u>, TIdx>
            {
                return idx.m_gridBlockIdx;
            }
        };

        //#############################################################################
        //! The IdxGbLinear grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbLinear<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka
