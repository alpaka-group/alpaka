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

#include <alpaka/accs/fibers/Common.hpp>

#include <alpaka/traits/Idx.hpp>        // idx::getIdx

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <map>                          // std::map

namespace alpaka
{
    namespace accs
    {
        namespace fibers
        {
            namespace detail
            {
                //#############################################################################
                //! The fibers accelerator index provider.
                //#############################################################################
                template<
                    typename TDim>
                class IdxFibers
                {
                public:
                    using FiberIdToIdxMap = std::map<boost::fibers::fiber::id, Vec<TDim>>;

                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxFibers(
                        FiberIdToIdxMap const & mFibersToIndices,
                        Vec<TDim> const & vuiGridBlockIdx) :
                        m_mFibersToIndices(mFibersToIndices),
                        m_vuiGridBlockIdx(vuiGridBlockIdx)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxFibers(IdxFibers const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA IdxFibers(IdxFibers &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxFibers const &) -> IdxFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxFibers &&) -> IdxFibers & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~IdxFibers() = default;

                    //-----------------------------------------------------------------------------
                    //! \return The index of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxBlockThread() const
                    -> Vec<TDim>
                    {
                        auto const idFiber(boost::this_fiber::get_id());
                        auto const itFind(m_mFibersToIndices.find(idFiber));
                        assert(itFind != m_mFibersToIndices.end());

                        return itFind->second;
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The index of the block of the currently executed thread.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdxGridBlock() const
                    -> Vec<TDim>
                    {
                        return m_vuiGridBlockIdx;
                    }

                private:
                    FiberIdToIdxMap const & m_mFibersToIndices;         //!< The mapping of fibers id's to fibers indices.
                    alignas(16u) Vec<TDim> const & m_vuiGridBlockIdx;   //!< The index of the currently executed block.
                };
            }
        }
    }

    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The CPU fibers accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                accs::fibers::detail::IdxFibers<TDim>>
            {
                using type = TDim;
            };
        }

        namespace idx
        {
            //#############################################################################
            //! The CPU fibers accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::fibers::detail::IdxFibers<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::fibers::detail::IdxFibers<TDim> const & index,
                    TWorkDiv const & workDiv)
                -> alpaka::Vec<TDim>
                {
                    boost::ignore_unused(workDiv);
                    return index.getIdxBlockThread();
                }
            };

            //#############################################################################
            //! The CPU fibers accelerator grid block index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                accs::fibers::detail::IdxFibers<TDim>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    accs::fibers::detail::IdxFibers<TDim> const & index,
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
