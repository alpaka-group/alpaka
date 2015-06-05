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

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST
#include <alpaka/core/Vec.hpp>          // Vec

#include <alpaka/traits/WorkDiv.hpp>    // GetWorkDiv

#include <iosfwd>                       // std::ostream

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! A basic class holding the work division as grid block extents and block thread extents.
        //#############################################################################
        template<
            typename TDim>
        class BasicWorkDiv
        {
        public:
            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            //ALPAKA_FCT_HOST BasicWorkDiv() = delete;
            //-----------------------------------------------------------------------------
            //! Constructor from values.
            //-----------------------------------------------------------------------------
            template<
                typename TGridBlockExtents,
                typename TBlockThreadExtents>
            ALPAKA_FCT_HOST_ACC explicit BasicWorkDiv(
                TGridBlockExtents const & gridBlockExtents = TGridBlockExtents(),
                TBlockThreadExtents const & blockThreadExtents = TBlockThreadExtents()) :
                m_vuiGridBlockExtents(extent::getExtentsVecNd<TDim, UInt>(gridBlockExtents)),
                m_vuiBlockThreadExtents(extent::getExtentsVecNd<TDim, UInt>(blockThreadExtents))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC explicit BasicWorkDiv(
                BasicWorkDiv<TDim> const & other) :
                    m_vuiGridBlockExtents(other.m_vuiGridBlockExtents),
                    m_vuiBlockThreadExtents(other.m_vuiBlockThreadExtents)
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST_ACC explicit BasicWorkDiv(
                TWorkDiv const & other) :
                    m_vuiGridBlockExtents(alpaka::subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other))),
                    m_vuiBlockThreadExtents(alpaka::subVecEnd<TDim>(getWorkDiv<Block, Threads>(other)))
            {}
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC BasicWorkDiv(BasicWorkDiv<TDim> &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto operator=(BasicWorkDiv<TDim> const &) -> BasicWorkDiv<TDim> & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto operator=(BasicWorkDiv<TDim> &&) -> BasicWorkDiv<TDim> & = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST_ACC auto operator=(
                TWorkDiv const & other)
            -> BasicWorkDiv<TDim> &
            {
                m_vuiGridBlockExtents = alpaka::subVecEnd<TDim>(getWorkDiv<Grid, Blocks>(other));
                m_vuiBlockThreadExtents = alpaka::subVecEnd<TDim>(getWorkDiv<Block, Threads>(other));
                return *this;
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC /*virtual*/ ~BasicWorkDiv() noexcept = default;

            //-----------------------------------------------------------------------------
            //! \return The grid block extents of the currently executed thread.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto getGridBlockExtents() const
            -> Vec<TDim>
            {
                return m_vuiGridBlockExtents;
            }
            //-----------------------------------------------------------------------------
            //! \return The block threads extents of the currently executed thread.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST_ACC auto getBlockThreadExtents() const
            -> Vec<TDim>
            {
                return m_vuiBlockThreadExtents;
            }

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
            BasicWorkDiv<TDim> const & workDiv)
        -> std::ostream &
        {
            return (os
                << "{gridBlockExtents: " << workDiv.getGridBlockExtents()
                << ", blockThreadExtents: " << workDiv.getBlockThreadExtents()
                << "}");
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The BasicWorkDiv dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                alpaka::workdiv::BasicWorkDiv<TDim>>
            {
                using type = TDim;
            };
        }

        namespace workdiv
        {
            //#############################################################################
            //! The BasicWorkDiv block thread extents trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWorkDiv<
                alpaka::workdiv::BasicWorkDiv<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getWorkDiv(
                    alpaka::workdiv::BasicWorkDiv<TDim> const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    return workDiv.getBlockThreadExtents();
                }
            };

            //#############################################################################
            //! The BasicWorkDiv grid block extents trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetWorkDiv<
                alpaka::workdiv::BasicWorkDiv<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getWorkDiv(
                    alpaka::workdiv::BasicWorkDiv<TDim> const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    return workDiv.getGridBlockExtents();
                }
            };
        }
    }
}
