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

#include <alpaka/traits/WorkDiv.hpp>    // GetWorkDiv

#include <alpaka/core/Vec.hpp>          // Vec

#include <iosfwd>                       // std::ostream

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! A basic class holding the work division as grid block extents and block thread extents.
        //#############################################################################
        class BasicWorkDiv
        {
        public:
            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST BasicWorkDiv() = delete;
            //-----------------------------------------------------------------------------
            //! Constructor from values.
            //-----------------------------------------------------------------------------
            template<
                typename TGridBlockExtents,
                typename TBlockThreadExtents>
            ALPAKA_FCT_HOST
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            explicit
#endif
            BasicWorkDiv(
                TGridBlockExtents const & gridBlockExtent = TGridBlockExtents(),
                TBlockThreadExtents const & blockThreadExtents = TBlockThreadExtents()) :
                m_v3uiGridBlockExtents(Vec<3u>::fromExtents(gridBlockExtent)),
                m_v3uiBlockThreadExtents(Vec<3u>::fromExtents(blockThreadExtents))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            explicit
#endif
            BasicWorkDiv(
                BasicWorkDiv const & other) :
                    m_v3uiGridBlockExtents(getWorkDiv<Grid, Blocks, dim::Dim3>(other)),
                    m_v3uiBlockThreadExtents(getWorkDiv<Block, Threads, dim::Dim3>(other))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            explicit
#endif
            BasicWorkDiv(
                TWorkDiv const & other) :
                    m_v3uiGridBlockExtents(getWorkDiv<Grid, Blocks, dim::Dim3>(other)),
                    m_v3uiBlockThreadExtents(getWorkDiv<Block, Threads, dim::Dim3>(other))
            {}
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST BasicWorkDiv(BasicWorkDiv &&) = default;
#endif
            //-----------------------------------------------------------------------------
            //! Copy assignment.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(
                BasicWorkDiv const & other)
            -> BasicWorkDiv &
            {
                m_v3uiGridBlockExtents = getWorkDiv<Grid, Blocks, dim::Dim3>(other);
                m_v3uiBlockThreadExtents = getWorkDiv<Block, Threads, dim::Dim3>(other);
                return *this;
            }
            //-----------------------------------------------------------------------------
            //! Copy assignment.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST auto operator=(
                TWorkDiv const & other)
            -> BasicWorkDiv &
            {
                m_v3uiGridBlockExtents = getWorkDiv<Grid, Blocks, dim::Dim3>(other);
                m_v3uiBlockThreadExtents = getWorkDiv<Block, Threads, dim::Dim3>(other);
                return *this;
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            //ALPAKA_FCT_HOST virtual ~BasicWorkDiv() noexcept = default;

            //-----------------------------------------------------------------------------
            //! \return The grid block extents of the currently executed thread.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto getGridBlockExtents() const
            -> Vec<3u>
            {
                return m_v3uiGridBlockExtents;
            }
            //-----------------------------------------------------------------------------
            //! \return The block threads extents of the currently executed thread.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto getBlockThreadExtents() const
            -> Vec<3u>
            {
                return m_v3uiBlockThreadExtents;
            }

        public:
            Vec<3u> m_v3uiGridBlockExtents;
            Vec<3u> m_v3uiBlockThreadExtents;
        };

        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST auto operator<<(
            std::ostream & os,
            BasicWorkDiv const & workDiv)
        -> std::ostream &
        {
            return (os
                << "{GridBlockExtents: " << workDiv.getGridBlockExtents()
                << ", BlockThreadExtents: " << workDiv.getBlockThreadExtents()
                << "}");
        }
    }

    namespace traits
    {
        namespace workdiv
        {
            //#############################################################################
            //! The work div block thread 3D extents trait specialization.
            //#############################################################################
            template<>
            struct GetWorkDiv<
                alpaka::workdiv::BasicWorkDiv,
                origin::Block,
                unit::Threads,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getWorkDiv(
                    alpaka::workdiv::BasicWorkDiv const & workDiv)
                -> alpaka::DimToVecT<alpaka::dim::Dim3>
                {
                    return workDiv.getBlockThreadExtents();
                }
            };

            //#############################################################################
            //! The work div grid block 3D extents trait specialization.
            //#############################################################################
            template<>
            struct GetWorkDiv<
                alpaka::workdiv::BasicWorkDiv,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static auto getWorkDiv(
                    alpaka::workdiv::BasicWorkDiv const & workDiv)
                -> alpaka::DimToVecT<alpaka::dim::Dim3>
                {
                    return workDiv.getGridBlockExtents();
                }
            };
        }
    }
}
