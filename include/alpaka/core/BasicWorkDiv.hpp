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

#include <iosfwd>						// std::ostream

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! A basic class holding the work division as grid blocks extents and block kernels extents.
        //#############################################################################
        class BasicWorkDiv
        {
        public:
            //-----------------------------------------------------------------------------
            //! Default constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST BasicWorkDiv() = default;
            //-----------------------------------------------------------------------------
            //! Constructor from values.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
            explicit 
#endif
            BasicWorkDiv(
                Vec<3u> const & v3uiGridBlocksExtent, 
                Vec<3u> const & v3uiBlockKernelsExtents) :
                m_v3uiGridBlocksExtents(v3uiGridBlocksExtent),
                m_v3uiBlockKernelsExtents(v3uiBlockKernelsExtents)
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
                    m_v3uiGridBlocksExtents(getWorkDiv<Grid, Blocks, dim::Dim3>(other)),
                    m_v3uiBlockKernelsExtents(getWorkDiv<Block, Kernels, dim::Dim3>(other))
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
                    m_v3uiGridBlocksExtents(getWorkDiv<Grid, Blocks, dim::Dim3>(other)),
                    m_v3uiBlockKernelsExtents(getWorkDiv<Block, Kernels, dim::Dim3>(other))
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
            ALPAKA_FCT_HOST BasicWorkDiv & operator=(
                BasicWorkDiv const & other)
            {
                m_v3uiGridBlocksExtents = getWorkDiv<Grid, Blocks, dim::Dim3>(other);
                m_v3uiBlockKernelsExtents = getWorkDiv<Block, Kernels, dim::Dim3>(other);
                return *this;
            }
            //-----------------------------------------------------------------------------
            //! Copy assignment.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FCT_HOST BasicWorkDiv & operator=(
            TWorkDiv const & other)
            {
                m_v3uiGridBlocksExtents = getWorkDiv<Grid, Blocks, dim::Dim3>(other);
                m_v3uiBlockKernelsExtents = getWorkDiv<Block, Kernels, dim::Dim3>(other);
                return *this;
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            //ALPAKA_FCT_HOST virtual ~BasicWorkDiv() noexcept = default;

            //-----------------------------------------------------------------------------
            //! \return The grid blocks extents of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Vec<3u> getGridBlocksExtents() const
            {
                return m_v3uiGridBlocksExtents;
            }
            //-----------------------------------------------------------------------------
            //! \return The block kernels extents of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Vec<3u> getBlockKernelsExtents() const
            {
                return m_v3uiBlockKernelsExtents;
            }

        private:
            Vec<3u> m_v3uiGridBlocksExtents;
            Vec<3u> m_v3uiBlockKernelsExtents;
        };

        //-----------------------------------------------------------------------------
        //! Stream out operator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST std::ostream & operator<<(
            std::ostream & os,
            BasicWorkDiv const & workDiv)
        {
            return (os << "{GridBlocks: " << workDiv.getGridBlocksExtents() << ", BlockKernels: " << workDiv.getBlockKernelsExtents() << "}");
        }
    }

    namespace traits
    {
        namespace workdiv
        {
            //#############################################################################
            //! The work div block kernels 3D extents trait specialization.
            //#############################################################################
            template<>
            struct GetWorkDiv<
                alpaka::workdiv::BasicWorkDiv,
                origin::Block,
                unit::Kernels,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of kernels in each dimension of a block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getWorkDiv(
                    alpaka::workdiv::BasicWorkDiv const & workDiv)
                {
                    return workDiv.getBlockKernelsExtents();
                }
            };

            //#############################################################################
            //! The work div grid blocks 3D extents trait specialization.
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
                ALPAKA_FCT_HOST_ACC static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getWorkDiv(
                    alpaka::workdiv::BasicWorkDiv const & workDiv)
                {
                    return workDiv.getGridBlocksExtents();
                }
            };
        }
    }
}
