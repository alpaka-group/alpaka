/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/idx/MapIdx.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/workdiv/Traits.hpp"

#ifdef _OPENMP

#    include <omp.h>

namespace alpaka
{
    namespace bt
    {
        //! The OpenMP accelerator index provider.
        template<typename TDim, typename TIdx>
        class IdxBtOmp : public concepts::Implements<ConceptIdxBt, IdxBtOmp<TDim, TIdx>>
        {
        };
    } // namespace bt

    namespace trait
    {
        //! The OpenMP accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtOmp<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The OpenMP accelerator block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtOmp<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static auto getIdx(bt::IdxBtOmp<TDim, TIdx> const& /* idx */, TWorkDiv const& workDiv) -> Vec<TDim, TIdx>
            {
                // We assume that the thread id is positive.
                ALPAKA_ASSERT_OFFLOAD(::omp_get_thread_num() >= 0);
                // \TODO: Would it be faster to precompute the index and cache it inside an array?
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num())),
                    getWorkDiv<Block, Threads>(workDiv));
            }
        };

        template<typename TIdx>
        struct GetIdx<bt::IdxBtOmp<DimInt<1u>, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static auto getIdx(bt::IdxBtOmp<DimInt<1u>, TIdx> const& /* idx */, TWorkDiv const&)
                -> Vec<DimInt<1u>, TIdx>
            {
                return Vec<DimInt<1u>, TIdx>(static_cast<TIdx>(::omp_get_thread_num()));
            }
        };

        //! The OpenMP accelerator block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtOmp<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka

#endif
