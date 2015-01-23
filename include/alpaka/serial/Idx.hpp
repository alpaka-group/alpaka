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

#include <alpaka/traits/Idx.hpp>  // idx::getIdx

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            //#############################################################################
            //! This serial accelerator index provider.
            //#############################################################################
            class IdxSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxSerial(
                    Vec<3u> const & v3uiGridBlockIdx) :
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxSerial(IdxSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxSerial(IdxSerial &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxSerial & operator=(IdxSerial const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxSerial() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxBlockKernel() const
                {
                    return {0, 0, 0};
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                Vec<3u> const & m_v3uiGridBlockIdx;
            };
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The serial accelerator 3D block kernels index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                serial::detail::IdxSerial,
                origin::Block,
                unit::Kernels,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current kernel in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                    serial::detail::IdxSerial const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxBlockKernel();
                }
            };

            //#############################################################################
            //! The serial accelerator 3D grid blocks index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                serial::detail::IdxSerial,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                    serial::detail::IdxSerial const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
