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

#include <alpaka/workdiv/Traits.hpp>                // GetWorkDiv

#include <alpaka/core/Vec.hpp>                      // Vec
#include <alpaka/core/Common.hpp>                   // ALPAKA_FCT_HOST

#include <iosfwd>                                   // std::ostream

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! A basic class holding the work division as grid block extents and block thread extents.
        //#############################################################################
        template<
            typename TDim>
        class WorkDivMembers
        {
        public:
            using WorkDivBase = WorkDivMembers;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            //ALPAKA_FCT_HOST WorkDivMembers() = delete;
            //-----------------------------------------------------------------------------
            //! Constructor from values.
            //-----------------------------------------------------------------------------
            template<
                typename TGridBlockExtents,
                typename TBlockThreadExtents>
            ALPAKA_FCT_HOST_ACC explicit WorkDivMembers(
                TGridBlockExtents const & gridBlockExtents = TGridBlockExtents(),
                TBlockThreadExtents const & blockThreadExtents = TBlockThreadExtents()) :
                m_vuiGridBlockExtents(extent::getExtentsVecNd<TDim, Uint>(gridBlockExtents)),
                m_vuiBlockThreadExtents(extent::getExtentsVecNd<TDim, Uint>(blockThreadExtents))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC explicit WorkDivMembers(
                WorkDivMembers<TDim> const & other) :
                    m_vuiGridBlockExtents(other.m_vuiGridBlockExtents),
                    m_vuiBlockThreadExtents(other.m_vuiBlockThreadExtents)
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST_ACC explicit WorkDivMembers(
                TWorkDiv const & other) :
                    m_vuiGridBlockExtents(alpaka::subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other))),
                    m_vuiBlockThreadExtents(alpaka::subVecEnd<TDim>(getWorkDiv<Block, Threads>(other)))
            {}
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC WorkDivMembers(WorkDivMembers<TDim> &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto operator=(WorkDivMembers<TDim> const &) -> WorkDivMembers<TDim> & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto operator=(WorkDivMembers<TDim> &&) -> WorkDivMembers<TDim> & = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST_ACC auto operator=(
                TWorkDiv const & other)
            -> WorkDivMembers<TDim> &
            {
                m_vuiGridBlockExtents = alpaka::subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other));
                m_vuiBlockThreadExtents = alpaka::subVecEnd<TDim>(getWorkDiv<Block, Threads>(other));
                return *this;
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC /*virtual*/ ~WorkDivMembers() = default;

        public:
            Vec<TDim> m_vuiGridBlockExtents;
            Vec<TDim> m_vuiBlockThreadExtents;
        };

        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        template<
            typename TDim>
        ALPAKA_FCT_HOST auto operator<<(
            std::ostream & os,
            WorkDivMembers<TDim> const & workDiv)
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
                typename TDim>
            struct DimType<
                workdiv::WorkDivMembers<TDim>>
            {
                using type = TDim;
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
                typename TDim>
            struct GetWorkDiv<
                WorkDivMembers<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getWorkDiv(
                    WorkDivMembers<TDim> const & workDiv)
                -> Vec<TDim>
                {
                    return workDiv.m_vuiBlockThreadExtents;
                }
            };

            //#############################################################################
            //! The WorkDivMembers grid block extents trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWorkDiv<
                WorkDivMembers<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getWorkDiv(
                    WorkDivMembers<TDim> const & workDiv)
                -> Vec<TDim>
                {
                    return workDiv.m_vuiGridBlockExtents;
                }
            };
        }
    }
}
