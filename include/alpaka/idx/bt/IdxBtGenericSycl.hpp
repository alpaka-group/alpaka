/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Luca Ferragina <luca.ferragina@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Aurora Perego <aurora.perego@cern.ch>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka::bt
{
    //! The SYCL accelerator ND index provider.
    template<typename TDim, typename TIdx>
    class IdxBtGenericSycl : public concepts::Implements<ConceptIdxBt, IdxBtGenericSycl<TDim, TIdx>>
    {
    public:
        using IdxBtBase = IdxBtGenericSycl;

        explicit IdxBtGenericSycl(sycl::nd_item<TDim::value> work_item) : m_item_bt{work_item}
        {
        }

        sycl::nd_item<TDim::value> m_item_bt;
    };
} // namespace alpaka::bt

namespace alpaka::trait
{
    //! The SYCL accelerator index dimension get trait specialization.
    template<typename TDim, typename TIdx>
    struct DimType<bt::IdxBtGenericSycl<TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The SYCL accelerator block thread index get trait specialization.
    template<typename TDim, typename TIdx>
    struct GetIdx<bt::IdxBtGenericSycl<TDim, TIdx>, origin::Block, unit::Threads>
    {
        //! \return The index of the current thread in the block.
        template<typename TWorkDiv>
        static auto getIdx(bt::IdxBtGenericSycl<TDim, TIdx> const& idx, TWorkDiv const&) -> Vec<TDim, TIdx>
        {
            if constexpr(TDim::value == 1)
                return Vec<TDim, TIdx>{static_cast<TIdx>(idx.m_item_bt.get_local_id(0))};
            else if constexpr(TDim::value == 2)
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(1)),
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(0))};
            }
            else
            {
                return Vec<TDim, TIdx>{
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(2)),
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(1)),
                    static_cast<TIdx>(idx.m_item_bt.get_local_id(0))};
            }
        }
    };

    //! The SYCL accelerator block thread index idx type trait specialization.
    template<typename TDim, typename TIdx>
    struct IdxType<bt::IdxBtGenericSycl<TDim, TIdx>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait

#endif
