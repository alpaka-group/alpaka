/* Copyright 2023 Jan Stephan, Aurora Perego
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

namespace alpaka::gb
{
    //! The SYCL accelerator ND index provider.
    template<typename TDim, typename TIdx>
    class IdxGbGenericSycl : public concepts::Implements<ConceptIdxGb, IdxGbGenericSycl<TDim, TIdx>>
    {
    public:
        using IdxGbBase = IdxGbGenericSycl;

        IdxGbGenericSycl() = default;
    };
} // namespace alpaka::gb

namespace alpaka::trait
{
    //! The SYCL accelerator index dimension get trait specialization.
    template<typename TDim, typename TIdx>
    struct DimType<gb::IdxGbGenericSycl<TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The SYCL accelerator grid block index get trait specialization.
    template<typename TDim, typename TIdx>
    struct GetIdx<gb::IdxGbGenericSycl<TDim, TIdx>, origin::Grid, unit::Blocks>
    {
        //! \return The index of the current block in the grid.
        template<typename TWorkDiv>
        static auto getIdx(gb::IdxGbGenericSycl<TDim, TIdx> const&, TWorkDiv const&)
        {
            auto const item = sycl::ext::oneapi::experimental::this_nd_item<TDim::value>();

            if constexpr(TDim::value == 1)
                return Vec<TDim, TIdx>(static_cast<TIdx>(item.get_group(0)));
            else if constexpr(TDim::value == 2)
            {
                return Vec<TDim, TIdx>(static_cast<TIdx>(item.get_group(1)), static_cast<TIdx>(item.get_group(0)));
            }
            else
            {
                return Vec<TDim, TIdx>(
                    static_cast<TIdx>(item.get_group(2)),
                    static_cast<TIdx>(item.get_group(1)),
                    static_cast<TIdx>(item.get_group(0)));
            }
        }
    };

    //! The SYCL accelerator grid block index idx type trait specialization.
    template<typename TDim, typename TIdx>
    struct IdxType<gb::IdxGbGenericSycl<TDim, TIdx>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait

#endif
