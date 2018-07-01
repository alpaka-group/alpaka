/**
* \file
* Copyright 2018 Benjamin Worpitz
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

#include <alpaka/idx/Traits.hpp>

#include <alpaka/dim/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! A block thread index.
            template<
                typename TDim,
                typename TSize>
            class IdxBtRef
            {
            public:
                using IdxBtBase = IdxBtRef;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtRef(
                    Vec<TDim, TSize> const & blockThreadIdx) :
                        m_blockThreadIdx(blockThreadIdx)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtRef(IdxBtRef const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtRef(IdxBtRef &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtRef const &) -> IdxBtRef & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtRef &&) -> IdxBtRef & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~IdxBtRef() = default;

            public:
                Vec<TDim, TSize> const & m_blockThreadIdx;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The IdxBtRef block thread index dimension get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                idx::bt::IdxBtRef<TDim, TSize>>
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
            //! The IdxBtRef block thread index grid block index get trait specialization.
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::bt::IdxBtRef<TDim, TSize>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_NO_CUDA static auto getIdx(
                    idx::bt::IdxBtRef<TDim, TSize> const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<TDim, TSize>
                {
                    boost::ignore_unused(workDiv);
                    return idx.m_blockThreadIdx;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The IdxBtRef block thread index indec type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtRef<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}
