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

#include <alpaka/workdiv/Traits.hpp>    // GetWorkDiv
#include <alpaka/size/Traits.hpp>       // size::Size

#include <alpaka/core/Vec.hpp>          // Vec
#include <alpaka/core/Common.hpp>       // ALPAKA_FN_HOST

#include <iosfwd>                       // std::ostream

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! A basic class holding the work division as grid block extents and block thread extents.
        //#############################################################################
        template<
            typename TDim,
            typename TSize>
        class WorkDivMembers
        {
        public:
            using WorkDivBase = WorkDivMembers;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            //ALPAKA_FN_HOST WorkDivMembers() = delete;
            //-----------------------------------------------------------------------------
            //! Constructor from values.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TGridBlockExtents,
                typename TBlockThreadExtents>
            ALPAKA_FN_HOST_ACC explicit WorkDivMembers(
                TGridBlockExtents const & gridBlockExtents = TGridBlockExtents(),
                TBlockThreadExtents const & blockThreadExtents = TBlockThreadExtents()) :
                m_vuiGridBlockExtents(extent::getExtentsVecEnd<TDim>(gridBlockExtents)),
                m_vuiBlockThreadExtents(extent::getExtentsVecEnd<TDim>(blockThreadExtents))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC explicit WorkDivMembers(
                WorkDivMembers const & other) :
                    m_vuiGridBlockExtents(other.m_vuiGridBlockExtents),
                    m_vuiBlockThreadExtents(other.m_vuiBlockThreadExtents)
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST_ACC explicit WorkDivMembers(
                TWorkDiv const & other) :
                    m_vuiGridBlockExtents(subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other))),
                    m_vuiBlockThreadExtents(subVecEnd<TDim>(getWorkDiv<Block, Threads>(other)))
            {}
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC WorkDivMembers(WorkDivMembers &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto operator=(WorkDivMembers const &) -> WorkDivMembers & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC auto operator=(WorkDivMembers &&) -> WorkDivMembers & = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST_ACC auto operator=(
                TWorkDiv const & other)
            -> WorkDivMembers<TDim, TSize> &
            {
                m_vuiGridBlockExtents = subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other));
                m_vuiBlockThreadExtents = subVecEnd<TDim>(getWorkDiv<Block, Threads>(other));
                return *this;
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_HOST_ACC /*virtual*/ ~WorkDivMembers() = default;

        public:
            Vec<TDim, TSize> m_vuiGridBlockExtents;
            Vec<TDim, TSize> m_vuiBlockThreadExtents;
        };

        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST auto operator<<(
            std::ostream & os,
            WorkDivMembers<TDim, TSize> const & workDiv)
        -> std::ostream &
        {
            return (os
                << "{gridBlockExtents: " << workDiv.m_vuiGridBlockExtents
                << ", blockThreadExtents: " << workDiv.m_vuiBlockThreadExtents
                << "}");
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The WorkDivMembers dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                workdiv::WorkDivMembers<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The WorkDivMembers size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                workdiv::WorkDivMembers<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The WorkDivMembers block thread extents trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivMembers<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    WorkDivMembers<TDim, TSize> const & workDiv)
                -> Vec<TDim, TSize>
                {
                    return workDiv.m_vuiBlockThreadExtents;
                }
            };

            //#############################################################################
            //! The WorkDivMembers grid block extents trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetWorkDiv<
                WorkDivMembers<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getWorkDiv(
                    WorkDivMembers<TDim, TSize> const & workDiv)
                -> Vec<TDim, TSize>
                {
                    return workDiv.m_vuiGridBlockExtents;
                }
            };
        }
    }
}
