/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/dim/Traits.hpp>            // dim::DimType
#include <alpaka/dim/DimIntegralConst.hpp>  // dim::DimInt<N>
#include <alpaka/size/Traits.hpp>           // size::traits::SizeType
#include <alpaka/workdiv/Traits.hpp>        // workdiv::getWorkDiv

#include <alpaka/core/Positioning.hpp>      // origin::Grid/Blocks, unit::Blocks, unit::Threads
#include <alpaka/vec/Vec.hpp>               // Vec<N>
#include <alpaka/core/Common.hpp>           // ALPAKA_FN_HOST_ACC

#include <utility>                          // std::forward

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The index specifics.
    //-----------------------------------------------------------------------------
    namespace idx
    {
        //-----------------------------------------------------------------------------
        //! The index traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The index get trait.
            //#############################################################################
            template<
                typename TIdx,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetIdx;
        }

        //-----------------------------------------------------------------------------
        //! Get the indices requested.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TIdx,
            typename TWorkDiv>
        ALPAKA_FN_HOST_ACC auto getIdx(
            TIdx const & idx,
            TWorkDiv const & workDiv)
        -> Vec<dim::Dim<TWorkDiv>, size::Size<TIdx>>
        {
            return
                traits::GetIdx<
                    TIdx,
                    TOrigin,
                    TUnit>
                ::getIdx(
                    idx,
                    workDiv);
        }
        //-----------------------------------------------------------------------------
        //! Get the indices requested.
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOrigin,
            typename TUnit,
            typename TIdxWorkDiv>
        ALPAKA_FN_HOST_ACC auto getIdx(
            TIdxWorkDiv const & idxWorkDiv)
        -> Vec<dim::Dim<TIdxWorkDiv>, size::Size<TIdxWorkDiv>>
        {
            return
                traits::GetIdx<
                    TIdxWorkDiv,
                    TOrigin,
                    TUnit>
                ::getIdx(
                    idxWorkDiv,
                    idxWorkDiv);
        }

        namespace traits
        {
            //#############################################################################
            //! The grid block index get trait specialization for classes with IdxGbBase member type.
            //#############################################################################
            template<
                typename TIdxGb>
            struct GetIdx<
                TIdxGb,
                origin::Grid,
                unit::Blocks,
                typename std::enable_if<
                    std::is_base_of<typename TIdxGb::IdxGbBase, typename std::decay<TIdxGb>::type>::value
                    && (!std::is_same<typename TIdxGb::IdxGbBase, typename std::decay<TIdxGb>::type>::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST_ACC static auto getIdx(
                    TIdxGb const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<dim::Dim<typename TIdxGb::IdxGbBase>, size::Size<typename TIdxGb::IdxGbBase>>
                {
                    // Delegate the call to the base class.
                    return
                        idx::getIdx<
                            origin::Grid,
                            unit::Blocks>(
                                static_cast<typename TIdxGb::IdxGbBase const &>(idx),
                                workDiv);
                }
            };

            //#############################################################################
            //! The block thread index get trait specialization for classes with IdxBtBase member type.
            //#############################################################################
            template<
                typename TIdxBt>
            struct GetIdx<
                TIdxBt,
                origin::Block,
                unit::Threads,
                typename std::enable_if<
                    std::is_base_of<typename TIdxBt::IdxBtBase, typename std::decay<TIdxBt>::type>::value
                    && (!std::is_same<typename TIdxBt::IdxBtBase, typename std::decay<TIdxBt>::type>::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST_ACC static auto getIdx(
                    TIdxBt const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<dim::Dim<typename TIdxBt::IdxBtBase>, size::Size<typename TIdxBt::IdxBtBase>>
                {
                    // Delegate the call to the base class.
                    return
                        idx::getIdx<
                            origin::Block,
                            unit::Threads>(
                                static_cast<typename TIdxBt::IdxBtBase const &>(idx),
                                workDiv);
                }
            };

            //#############################################################################
            //! The grid thread index get trait specialization.
            //#############################################################################
            template<
                typename TIdx>
            struct GetIdx<
                TIdx,
                origin::Grid,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TWorkDiv>
                ALPAKA_FN_HOST_ACC static auto getIdx(
                    TIdx const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<dim::Dim<TIdx>, size::Size<TIdx>>
                {
                    return
                        idx::getIdx<origin::Grid, unit::Blocks>(idx, workDiv)
                        * workdiv::getWorkDiv<origin::Block, unit::Threads>(workDiv)
                        + idx::getIdx<origin::Block, unit::Threads>(idx, workDiv);
                }
            };
        }
    }
}
