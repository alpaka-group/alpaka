/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Omp5.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/idx/MapIdx.hpp>

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxBtOmp5BuiltIn : public concepts::Implements<ConceptIdxBt, IdxBtOmp5BuiltIn<TDim, TIdx>>
            {
            public:
                //-----------------------------------------------------------------------------
                IdxBtOmp5BuiltIn() = default;
                //-----------------------------------------------------------------------------
                IdxBtOmp5BuiltIn(IdxBtOmp5BuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxBtOmp5BuiltIn(IdxBtOmp5BuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtOmp5BuiltIn const & ) -> IdxBtOmp5BuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtOmp5BuiltIn &&) -> IdxBtOmp5BuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtOmp5BuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtOmp5BuiltIn<TDim, TIdx>>
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
            //! The GPU CUDA accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtOmp5BuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    idx::bt::IdxBtOmp5BuiltIn<TDim, TIdx> const &,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    // We assume that the thread id is positive.
                    ALPAKA_ASSERT(::omp_get_thread_num()>=0);
                    // \TODO: Would it be faster to precompute the index and cache it inside an array?
                    return idx::mapIdx<TDim::value>(
                        vec::Vec<dim::DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num())),
                        workdiv::getWorkDiv<Block, Threads>(workDiv));
                }
            };

            template<
                typename TIdx>
            struct GetIdx<
                idx::bt::IdxBtOmp5BuiltIn<dim::DimInt<1u>, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    idx::bt::IdxBtOmp5BuiltIn<dim::DimInt<1u>, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<dim::DimInt<1u>, TIdx>
                {
                    alpaka::ignore_unused(idx);
                    return vec::Vec<dim::DimInt<1u>, TIdx>(static_cast<TIdx>(omp_get_thread_num()));
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtOmp5BuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
