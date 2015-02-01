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
                ALPAKA_FCT_ACC_NO_CUDA IdxOpenMp & operator=(IdxOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxOpenMp() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //
                // \TODO: Would it be faster to precompute the 3 dimensional index and cache it inside an array?
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxBlockKernel3d(TWorkDiv const & workDiv) const
                {
                    // We assume that the thread id is positive.
                    auto const v1iIdxBlockKernel(getIdxBlockKernel1d());
                    // Get the number of kernels in each dimension of the grid.
                    auto const v3uiBlockKernelsExtents(workdiv::getWorkDiv<Block, Kernels, dim::Dim3>(workDiv));
                    auto const v2uiBlockKernelsExtents(v3uiBlockKernelsExtents.template subvec<2u>());

                    return mapIdx<3>(
                        v1iIdxBlockKernel,
                        v2uiBlockKernelsExtents);
                }
                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<1u> getIdxBlockKernel1d() const
                {
                    // We assume that the thread id is positive.
                    assert(::omp_get_thread_num() >= 0);
                    auto const uiThreadId(static_cast<Vec<1u>::Value>(::omp_get_thread_num()));
                    return Vec<1u>(uiThreadId);
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxGridBlock() const
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
            //! The OpenMP accelerator 3D block kernel index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                openmp::detail::IdxOpenMp,
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
                    openmp::detail::IdxOpenMp const & index,
                    TWorkDiv const & workDiv)
                {
                    return index.getIdxBlockKernel3d(workDiv);
                }
            };

            //#############################################################################
            //! The OpenMP accelerator 1D block kernel index get trait specialization.
            //#############################################################################
            template<>
            struct GetIdx<
                openmp::detail::IdxOpenMp,
                origin::Block,
                unit::Kernels,
                alpaka::dim::Dim1>
            {
                //-----------------------------------------------------------------------------
                //! \return The 3-dimensional index of the current kernel in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static alpaka::dim::DimToVecT<alpaka::dim::Dim1> getIdx(
                    openmp::detail::IdxOpenMp const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxBlockKernel1d();
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
                ALPAKA_FCT_ACC_NO_CUDA static alpaka::dim::DimToVecT<alpaka::dim::Dim3> getIdx(
                    openmp::detail::IdxOpenMp const & index,
                    TWorkDiv const &)
                {
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
