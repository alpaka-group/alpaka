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
    namespace accs
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
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxSerial(IdxSerial &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxSerial const &) -> IdxSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxSerial() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread() const
                    -> Vec<3u>
                    {
                        return Vec<3u>(0u, 0u, 0u);
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The block index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxGridBlock() const
                    -> Vec<3u>
                    {
                        return m_v3uiGridBlockIdx;
                    }

                private:
                    Vec<3u> const & m_v3uiGridBlockIdx;
                };
            }
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The serial accelerator 3D block thread index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                accs::serial::detail::IdxSerial,
                origin::Block,
                unit::Threads,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::serial::detail::IdxSerial const & index,
                    TWorkDiv const &)
                -> alpaka::DimToVecT<alpaka::dim::Dim3>
                {
                    return index.getIdxBlockThread();
                }
            };

            //#############################################################################
            //! The serial accelerator 3D grid block index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                accs::serial::detail::IdxSerial,
                origin::Grid,
                unit::Blocks,
                alpaka::dim::Dim3>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::serial::detail::IdxSerial const & index,
                    TWorkDiv const &)
                -> alpaka::DimToVecT<alpaka::dim::Dim3>
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
