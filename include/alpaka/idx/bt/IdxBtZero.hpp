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

#include <alpaka/dim/Traits.hpp>                // dim::DimT

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! A zero block thread index provider.
            //#############################################################################
            template<
                typename TDim>
            class IdxBtZero
            {
            public:
                using IdxBtBase = IdxBtZero;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtZero() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtZero(IdxBtZero const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtZero(IdxBtZero &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxBtZero const &) -> IdxBtZero & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxBtZero &&) -> IdxBtZero & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~IdxBtZero() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The zero block thread index provider dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                idx::bt::IdxBtZero<TDim>>
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
            //! The zero block thread index provider block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                bt::IdxBtZero<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    bt::IdxBtZero<TDim> const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<TDim>
                {
                    boost::ignore_unused(idx);
                    boost::ignore_unused(workDiv);
                    return Vec<TDim>::zeros();
                }
            };
        }
    }
}
