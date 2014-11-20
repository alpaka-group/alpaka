/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/openmp/WorkSize.hpp>   // TInterfacedWorkSize

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
            //! This class holds the implementation details for the indexing of the OpenMP accelerator.
            //#############################################################################
            class IndexOpenMp
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexOpenMp(
                    TInterfacedWorkSize const & workSize,
                    vec<3u> const & v3uiGridBlockIdx) :
                    m_WorkSize(workSize),
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexOpenMp(IndexOpenMp const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexOpenMp(IndexOpenMp &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexOpenMp & operator=(IndexOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~IndexOpenMp() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxBlockKernel() const
                {
                    // We assume that the thread id is positive.
                    assert(::omp_get_thread_num()>=0);
                    auto const uiThreadId(static_cast<std::uint32_t>(::omp_get_thread_num()));
                    // Get the number of kernels in each dimension of the grid.
                    auto const v3uiSizeBlockKernels(m_WorkSize.getSize<Block, Kernels, D3>());

                    return mapIndex<3>(
                        vec<1u>(uiThreadId), 
                        m_WorkSize.getSize<Block, Kernels, D3>().subvec<2>());
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                TInterfacedWorkSize const & m_WorkSize;        //!< The mapping of thread id's to thread indices.
                vec<3u> const & m_v3uiGridBlockIdx;        //!< The index of the currently executed block.
            };
            using TInterfacedIndex = alpaka::detail::IIndex<IndexOpenMp>;
        }
    }
}
