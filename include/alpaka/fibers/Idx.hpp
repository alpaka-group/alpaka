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

#include <alpaka/fibers/Common.hpp>

#include <alpaka/traits/Idx.hpp>    // idx::getIdx

#include <map>                      // std::map

namespace alpaka
{
    namespace fibers
    {
        namespace detail
        {
            using FiberIdToIdxMap = std::map<boost::fibers::fiber::id, Vec<3u>>;
            //#############################################################################
            //! This fibers accelerator index provider.
            //#############################################################################
            class IdxFibers
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxFibers(
                    FiberIdToIdxMap const & mFibersToIndices,
                    Vec<3u> const & v3uiGridBlockIdx) :
                    m_mFibersToIndices(mFibersToIndices),
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxFibers(IdxFibers const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxFibers(IdxFibers &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxFibers & operator=(IdxFibers const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxFibers() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxBlockKernel() const
                {
                    auto const idFiber(boost::this_fiber::get_id());
                    auto const itFind(m_mFibersToIndices.find(idFiber));
                    assert(itFind != m_mFibersToIndices.end());

                    return itFind->second;
                }
                //-----------------------------------------------------------------------------
                //! \return The index of the block of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                FiberIdToIdxMap const & m_mFibersToIndices;     //!< The mapping of fibers id's to fibers indices.
                Vec<3u> const & m_v3uiGridBlockIdx;             //!< The index of the currently executed block.
            };
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The fibers accelerator 3D block kernel index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                fibers::detail::IdxFibers,
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
                    fibers::detail::IdxFibers const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxBlockKernel();
                }
            };

            //#############################################################################
            //! The fibers accelerator 3D grid block index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                fibers::detail::IdxFibers,
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
                    fibers::detail::IdxFibers const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
