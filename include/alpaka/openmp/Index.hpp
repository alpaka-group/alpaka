/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/openmp/WorkExtent.hpp> // TInterfacedWorkExtent

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
                ALPAKA_FCT_HOST IndexOpenMp(
                    TInterfacedWorkExtent const & workExtent,
                    vec<3u> const & v3uiGridBlockIdx) :
                    m_WorkExtent(workExtent),
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
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST IndexOpenMp & operator=(IndexOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~IndexOpenMp() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //
                // \TODO: Would it be faster to precompute the 3 dimensional index and cache it inside an array?
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxBlockKernel() const
                {
                    // We assume that the thread id is positive.
                    assert(::omp_get_thread_num()>=0);
                    auto const uiThreadId(static_cast<std::uint32_t>(::omp_get_thread_num()));
                    // Get the number of kernels in each dimension of the grid.
                    auto const v3uiBlockKernelsExtent(m_WorkExtent.getExtent<Block, Kernels, D3>());

                    return mapIndex<3>(
                        vec<1u>(uiThreadId), 
                        m_WorkExtent.getExtent<Block, Kernels, D3>().subvec<2>());
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                TInterfacedWorkExtent const & m_WorkExtent;        //!< The mapping of thread id's to thread indices.
                vec<3u> const & m_v3uiGridBlockIdx;        //!< The index of the currently executed block.
            };
            using TInterfacedIndex = alpaka::detail::IIndex<IndexOpenMp>;
        }
    }
}
