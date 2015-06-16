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

#include <alpaka/idx/Traits.hpp>                // idx::GetIdx

#include <alpaka/core/OpenMp.hpp>
#include <alpaka/core/MapIdx.hpp>               // mapIdx

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The OpenMP accelerator index provider.
            //#############################################################################
            template<
                typename TDim>
            class IdxBtOmp
            {
            public:
                using IdxBtBase = IdxBtOmp;

                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtOmp() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtOmp(IdxBtOmp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA IdxBtOmp(IdxBtOmp &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxBtOmp const &) -> IdxBtOmp & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(IdxBtOmp &&) -> IdxBtOmp & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA /*virtual*/ ~IdxBtOmp() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenMP accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct DimType<
                idx::bt::IdxBtOmp<TDim>>
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
            //! The OpenMP accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct GetIdx<
                bt::IdxBtOmp<TDim>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA static auto getIdx(
                    bt::IdxBtOmp<TDim> const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<TDim>
                {
                    boost::ignore_unused(idx);
                    // We assume that the thread id is positive.
                    assert(::omp_get_thread_num()>=0);
                    // \TODO: Would it be faster to precompute the index and cache it inside an array?
                    return mapIdx<TDim::value>(
                        Vec1<>(static_cast<Vec1<>::Val>(::omp_get_thread_num())),
                        workdiv::getWorkDiv<Block, Threads>(workDiv));
                }
            };
        }
    }
}
