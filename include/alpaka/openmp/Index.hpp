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

#include <alpaka/openmp/WorkDiv.hpp> // InterfacedWorkDivOpenMp

#include <alpaka/openmp/Common.hpp>

#include <alpaka/interfaces/Index.hpp>  // IIndex

#include <alpaka/core/IndexMapping.hpp> // IIndex

namespace alpaka
{
    namespace openmp
    {
        namespace detail
        {
            //#############################################################################
            //! This OpenMP accelerator index provider.
            //#############################################################################
            class IndexOpenMp
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexOpenMp(
                    InterfacedWorkDivOpenMp const & workDiv,
                    Vec<3u> const & v3uiGridBlockIdx) :
                    m_WorkDiv(workDiv),
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexOpenMp(IndexOpenMp const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexOpenMp(IndexOpenMp &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexOpenMp & operator=(IndexOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~IndexOpenMp() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //
                // \TODO: Would it be faster to precompute the 3 dimensional index and cache it inside an array?
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxBlockKernel() const
                {
                    // We assume that the thread id is positive.
                    assert(::omp_get_thread_num()>=0);
                    auto const uiThreadId(static_cast<std::uint32_t>(::omp_get_thread_num()));
                    // Get the number of kernels in each dimension of the grid.
                    auto const v3uiBlockKernelsExtents(m_WorkDiv.getExtents<Block, Kernels, dim::Dim3>());

                    return mapIndex<3>(
                        Vec<1u>(uiThreadId), 
                        m_WorkDiv.getExtents<Block, Kernels, dim::Dim3>().subvec<2>());
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                InterfacedWorkDivOpenMp const & m_WorkDiv;        //!< The mapping of thread id's to thread indices.
                Vec<3u> const & m_v3uiGridBlockIdx;        //!< The index of the currently executed block.
            };
            using InterfacedIndexOpenMp = alpaka::detail::IIndex<IndexOpenMp>;
        }
    }
}
