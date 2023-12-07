/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

namespace alpaka
{
    namespace bt
    {
        //! A zero block thread index provider.
        template<typename TDim, typename TIdx>
        class IdxBtZero : public concepts::Implements<ConceptIdxBt, IdxBtZero<TDim, TIdx>>
        {
        };
    } // namespace bt

    namespace trait
    {
        //! The zero block thread index provider dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtZero<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The zero block thread index provider block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtZero<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            ALPAKA_FN_HOST static auto getIdx(
                bt::IdxBtZero<TDim, TIdx> const& /* idx */,
                TWorkDiv const& /* workDiv */) -> Vec<TDim, TIdx>
            {
                return Vec<TDim, TIdx>::zeros();
            }
        };

        //! The zero block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtZero<TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka
