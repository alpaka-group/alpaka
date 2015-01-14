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

#include <alpaka/interfaces/Index.hpp>  // IIndex

#include <thread>                       // std::thread
#include <map>                          // std::map

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            using ThreadIdToIndexMap = std::map<std::thread::id, Vec<3u>>;
            //#############################################################################
            //! This threads accelerator index provider.
            //#############################################################################
            class IndexThreads
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexThreads(
                    ThreadIdToIndexMap const & mThreadsToIndices,
                    Vec<3u> const & v3uiGridBlockIdx) :
                    m_mThreadsToIndices(mThreadsToIndices),
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexThreads(IndexThreads const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexThreads(IndexThreads &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IndexThreads & operator=(IndexThreads const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~IndexThreads() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxBlockKernel() const
                {
                    auto const idThread(std::this_thread::get_id());
                    auto const itFind(m_mThreadsToIndices.find(idThread));
                    assert(itFind != m_mThreadsToIndices.end());

                    return itFind->second;
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA Vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                ThreadIdToIndexMap const & m_mThreadsToIndices;   //!< The mapping of thread id's to thread indices.
                Vec<3u> const & m_v3uiGridBlockIdx;             //!< The index of the currently executed block.
            };
            using InterfacedIndexThreads = alpaka::detail::IIndex<IndexThreads>;
        }
    }
}
