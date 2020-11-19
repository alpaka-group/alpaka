/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/Traits.hpp>

#include <utility>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Get the indices requested.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOrigin, typename TUnit, typename TIdx, typename TWorkDiv>
    ALPAKA_FN_HOST_ACC auto getIdx(TIdx const& idx, TWorkDiv const& workDiv) -> Vec<Dim<TWorkDiv>, Idx<TIdx>>
    {
        return traits::GetIdx<TIdx, TOrigin, TUnit>::getIdx(idx, workDiv);
    }
    //-----------------------------------------------------------------------------
    //! Get the indices requested.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOrigin, typename TUnit, typename TIdxWorkDiv>
    ALPAKA_FN_HOST_ACC auto getIdx(TIdxWorkDiv const& idxWorkDiv) -> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        return traits::GetIdx<TIdxWorkDiv, TOrigin, TUnit>::getIdx(idxWorkDiv, idxWorkDiv);
    }

    namespace traits
    {
        //#############################################################################
        //! The grid block index get trait specialization for classes with IdxGbBase member type.
        template<typename TIdxGb>
        struct GetIdx<TIdxGb, origin::Grid, unit::Blocks>
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptIdxGb, TIdxGb>;
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TWorkDiv>
            ALPAKA_FN_HOST_ACC static auto getIdx(TIdxGb const& idx, TWorkDiv const& workDiv)
                -> Vec<Dim<ImplementationBase>, Idx<ImplementationBase>>
            {
                return traits::GetIdx<ImplementationBase, origin::Grid, unit::Blocks>::getIdx(idx, workDiv);
            }
        };

        //#############################################################################
        //! The block thread index get trait specialization for classes with IdxBtBase member type.
        template<typename TIdxBt>
        struct GetIdx<TIdxBt, origin::Block, unit::Threads>
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptIdxBt, TIdxBt>;
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TWorkDiv>
            ALPAKA_FN_HOST_ACC static auto getIdx(TIdxBt const& idx, TWorkDiv const& workDiv)
                -> Vec<Dim<ImplementationBase>, Idx<ImplementationBase>>
            {
                return traits::GetIdx<ImplementationBase, origin::Block, unit::Threads>::getIdx(idx, workDiv);
            }
        };

        //#############################################################################
        //! The grid thread index get trait specialization.
        template<typename TIdx>
        struct GetIdx<TIdx, origin::Grid, unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the grid.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TWorkDiv>
            ALPAKA_FN_HOST_ACC static auto getIdx(TIdx const& idx, TWorkDiv const& workDiv)
            {
                return alpaka::getIdx<origin::Grid, unit::Blocks>(idx, workDiv)
                    * getWorkDiv<origin::Block, unit::Threads>(workDiv)
                    + alpaka::getIdx<origin::Block, unit::Threads>(idx, workDiv);
            }
        };
    } // namespace traits
    //-----------------------------------------------------------------------------
    //! Get the index of the first element this thread computes.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIdxWorkDiv, typename TGridThreadIdx, typename TThreadElemExtent>
    ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(
        TIdxWorkDiv const& idxWorkDiv,
        TGridThreadIdx const& gridThreadIdx,
        TThreadElemExtent const& threadElemExtent) -> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        alpaka::ignore_unused(idxWorkDiv);

        return gridThreadIdx * threadElemExtent;
    }
    //-----------------------------------------------------------------------------
    //! Get the index of the first element this thread computes.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIdxWorkDiv, typename TGridThreadIdx>
    ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(TIdxWorkDiv const& idxWorkDiv, TGridThreadIdx const& gridThreadIdx)
        -> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        auto const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(idxWorkDiv));
        return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx, threadElemExtent);
    }
    //-----------------------------------------------------------------------------
    //! Get the index of the first element this thread computes.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIdxWorkDiv>
    ALPAKA_FN_HOST_ACC auto getIdxThreadFirstElem(TIdxWorkDiv const& idxWorkDiv)
        -> Vec<Dim<TIdxWorkDiv>, Idx<TIdxWorkDiv>>
    {
        auto const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(idxWorkDiv));
        return getIdxThreadFirstElem(idxWorkDiv, gridThreadIdx);
    }
} // namespace alpaka
