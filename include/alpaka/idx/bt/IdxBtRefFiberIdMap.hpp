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

#include <alpaka/idx/Traits.hpp>                // idx::getIdx

#include <alpaka/core/Fibers.hpp>

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

#include <map>                                  // std::map

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The fibers accelerator index provider.
            //#############################################################################
            template<
                typename TDim>
            class IdxBtRefFiberIdMap
            {
            public:
                using IdxBtBase = IdxBtRefFiberIdMap;

                using FiberIdToIdxMap = std::map<boost::fibers::fiber::id, Vec<TDim>>;

                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtRefFiberIdMap(
                    FiberIdToIdxMap const & mFibersToIndices) :
                    m_mFibersToIndices(mFibersToIndices)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtRefFiberIdMap(IdxBtRefFiberIdMap const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtRefFiberIdMap(IdxBtRefFiberIdMap &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxBtRefFiberIdMap const &) -> IdxBtRefFiberIdMap & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxBtRefFiberIdMap &&) -> IdxBtRefFiberIdMap & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~IdxBtRefFiberIdMap() = default;

            public:
                FiberIdToIdxMap const & m_mFibersToIndices;         //!< The mapping of fibers id's to fibers indices.
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                idx::bt::IdxBtRefFiberIdMap<TDim>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                bt::IdxBtRefFiberIdMap<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    bt::IdxBtRefFiberIdMap<TDim> const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<TDim>
                {
                    boost::ignore_unused(workDiv);
                    auto const idFiber(boost::this_fiber::get_id());
                    auto const itFind(idx.m_mFibersToIndices.find(idFiber));
                    assert(itFind != idx.m_mFibersToIndices.end());
                    return itFind->second;
                }
            };
        }
    }
}
