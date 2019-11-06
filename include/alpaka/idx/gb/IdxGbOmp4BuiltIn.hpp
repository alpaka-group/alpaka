/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Omp4.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>

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
            class IdxGbOmp4BuiltIn
            {
            public:
                using IdxGbBase = IdxGbOmp4BuiltIn;

                static_assert(TDim::value == 1, "Omp4 only supports 1D grids.");

                //-----------------------------------------------------------------------------
                IdxGbOmp4BuiltIn() : m_gridBlockOffset(static_cast<TIdx>(0u)) {}
                IdxGbOmp4BuiltIn(const TIdx &gridBlockOffset) : m_gridBlockOffset(gridBlockOffset) {}
                //-----------------------------------------------------------------------------
                IdxGbOmp4BuiltIn(IdxGbOmp4BuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxGbOmp4BuiltIn(IdxGbOmp4BuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbOmp4BuiltIn const & ) -> IdxGbOmp4BuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbOmp4BuiltIn &&) -> IdxGbOmp4BuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbOmp4BuiltIn() = default;

                TIdx const m_gridBlockOffset;
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
                idx::gb::IdxGbOmp4BuiltIn<TDim, TIdx>>
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
                idx::gb::IdxGbOmp4BuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    idx::gb::IdxGbOmp4BuiltIn<TDim, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    return idx.m_gridBlockOffset + static_cast<TIdx>(omp_get_team_num());
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
                idx::gb::IdxGbOmp4BuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
