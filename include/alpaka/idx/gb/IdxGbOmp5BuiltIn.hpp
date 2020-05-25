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
        namespace gb
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxGbOmp5BuiltIn : public concepts::Implements<ConceptIdxGb, IdxGbOmp5BuiltIn<TDim, TIdx>>
            {
            public:
                //-----------------------------------------------------------------------------
                IdxGbOmp5BuiltIn(const TIdx &teamOffset = static_cast<TIdx>(0u)) : m_teamOffset(teamOffset) {}
                //-----------------------------------------------------------------------------
                IdxGbOmp5BuiltIn(IdxGbOmp5BuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxGbOmp5BuiltIn(IdxGbOmp5BuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbOmp5BuiltIn const & ) -> IdxGbOmp5BuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbOmp5BuiltIn &&) -> IdxGbOmp5BuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbOmp5BuiltIn() = default;

                TIdx const m_teamOffset; //! \todo what is this for?
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
                idx::gb::IdxGbOmp5BuiltIn<TDim, TIdx>>
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
            //! The GPU CUDA accelerator grid block index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::gb::IdxGbOmp5BuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    idx::gb::IdxGbOmp5BuiltIn<TDim, TIdx> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    // We assume that the thread id is positive.
                    ALPAKA_ASSERT(::omp_get_team_num()>=0);
                    // \TODO: Would it be faster to precompute the index and cache it inside an array?
                    return idx::mapIdx<TDim::value>(
                        vec::Vec<dim::DimInt<1u>, TIdx>(static_cast<TIdx>(idx.m_teamOffset + static_cast<TIdx>(::omp_get_team_num()))),
                        workdiv::getWorkDiv<Grid, Blocks>(workDiv));
                }
            };

            template<
                typename TIdx>
            struct GetIdx<
                idx::gb::IdxGbOmp5BuiltIn<dim::DimInt<1u>, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    idx::gb::IdxGbOmp5BuiltIn<dim::DimInt<1u>, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<dim::DimInt<1u>, TIdx>
                {
                    return static_cast<TIdx>(idx.m_teamOffset + static_cast<TIdx>(omp_get_team_num()));
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator grid block index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::gb::IdxGbOmp5BuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
