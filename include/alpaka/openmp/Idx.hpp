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

#include <alpaka/openmp/WorkDiv.hpp>    // WorkDivOpenMp

#include <alpaka/openmp/Common.hpp>

#include <alpaka/traits/Idx.hpp>        // idx::GetIdx

#include <alpaka/core/IdxMapping.hpp>   // mapIdx

namespace alpaka
{
    namespace openmp
    {
        namespace detail
        {
            //#############################################################################
            //! This OpenMP accelerator index provider.
            //#############################################################################
            class IdxOpenMp
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxOpenMp(
                    Vec<3u> const & v3uiGridBlockIdx) :
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxOpenMp(IdxOpenMp const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxOpenMp(IdxOpenMp &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxOpenMp const &) -> IdxOpenMp & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxOpenMp() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed thread.
                //
                // \TODO: Would it be faster to precompute the 3 dimensional index and cache it inside an array?
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread3d(TWorkDiv const & workDiv) const
                -> Vec<3u>
                {
                    // We assume that the thread id is positive.
                    auto const v1iIdxBlockThread(getIdxBlockThread1d());
                    // Get the number of threads in each dimension of the grid.
                    auto const v3uiBlockThreadExtents(workdiv::getWorkDiv<Block, Threads, dim::Dim3>(workDiv));
                    auto const v2uiBlockThreadExtents(v3uiBlockThreadExtents.template subVec<2u>());

                    return mapIdx<3>(
                        v1iIdxBlockThread,
                        v2uiBlockThreadExtents);
                }
                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed thread.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread1d() const
                -> Vec<1u>
                {
                    // We assume that the thread id is positive.
                    assert(::omp_get_thread_num() >= 0);
                    auto const uiThreadId(static_cast<Vec<1u>::Val>(::omp_get_thread_num()));
                    return Vec<1u>(uiThreadId);
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
                Vec<3u> const & m_v3uiGridBlockIdx; //!< The index of the currently executed block.
            };
        }
    }

    namespace traits
    {
        namespace idx
        {
            //#############################################################################
            //! The OpenMP accelerator 3D block thread index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                openmp::detail::IdxOpenMp,
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
                    openmp::detail::IdxOpenMp const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::DimToVecT<alpaka::dim::Dim3>
                {
                    return index.getIdxBlockThread3d(workDiv);
                }
            };

            //#############################################################################
            //! The OpenMP accelerator 1D block thread index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                openmp::detail::IdxOpenMp,
                origin::Block,
                unit::Threads,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    openmp::detail::IdxOpenMp const & index,
                    TWorkDiv const &)
                -> alpaka::DimToVecT<alpaka::dim::Dim1>
                {
                    return index.getIdxBlockThread1d();
                }
            };

            //#############################################################################
            //! The OpenMP accelerator 3D grid block index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                openmp::detail::IdxOpenMp,
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
                    openmp::detail::IdxOpenMp const & index,
                    TWorkDiv const &)
                -> alpaka::DimToVecT<alpaka::dim::Dim3>
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
