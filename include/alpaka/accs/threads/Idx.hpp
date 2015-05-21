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

#include <alpaka/traits/Idx.hpp>        // idx::getIdx

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <thread>                       // std::thread
#include <map>                          // std::map

namespace alpaka
{
    namespace accs
    {
        namespace threads
        {
            namespace detail
            {
                //#############################################################################
                //! This threads accelerator index provider.
                //#############################################################################
                template<
                    typename TDim>
                class IdxThreads
                {
                public:
                    using ThreadIdToIdxMap = std::map<std::thread::id, Vec<TDim>>;

                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxThreads(
                        ThreadIdToIdxMap const & mThreadsToIndices,
                        Vec<TDim> const & vuiGridBlockIdx) :
                        m_mThreadsToIndices(mThreadsToIndices),
                        m_vuiGridBlockIdx(vuiGridBlockIdx)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxThreads(IdxThreads const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxThreads(IdxThreads &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxThreads const &) -> IdxThreads & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxThreads() = default;
#else
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~IdxThreads() noexcept = default;
#endif

                    //-----------------------------------------------------------------------------
                    //! \return The index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread() const
                    -> Vec<TDim>
                    {
                        auto const idThread(std::this_thread::get_id());
                        auto const itFind(m_mThreadsToIndices.find(idThread));
                        assert(itFind != m_mThreadsToIndices.end());

                        return itFind->second;
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The block index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxGridBlock() const
                    -> Vec<TDim>
                    {
                        return m_vuiGridBlockIdx;
                    }

                private:
                    ThreadIdToIdxMap const & m_mThreadsToIndices;   //!< The mapping of thread id's to thread indices.
                    Vec<TDim> const & m_vuiGridBlockIdx;            //!< The index of the currently executed block.
                };
            }
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The CPU threads accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::threads::detail::IdxThreads<TDim>>
            {
                using type = TDim;
            };
        }

        namespace idx
        {
            //#############################################################################
            //! The CPU threads accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::threads::detail::IdxThreads<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::threads::detail::IdxThreads<TDim> const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    boost::ignore_unused(workDiv);
                    return index.getIdxBlockThread();
                }
            };

            //#############################################################################
            //! The CPU threads accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::threads::detail::IdxThreads<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::threads::detail::IdxThreads<TDim> const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    boost::ignore_unused(workDiv);
                    return index.getIdxGridBlock();
                }
            };
        }
    }
}
