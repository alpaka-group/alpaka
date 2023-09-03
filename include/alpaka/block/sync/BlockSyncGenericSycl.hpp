/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/sync/Traits.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL block synchronization.
    template<typename TDim>
    class BlockSyncGenericSycl : public concepts::Implements<ConceptBlockSync, BlockSyncGenericSycl<TDim>>
    {
    public:
        using BlockSyncBase = BlockSyncGenericSycl<TDim>;

        BlockSyncGenericSycl() = default;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename TDim>
    struct SyncBlockThreads<BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreads(BlockSyncGenericSycl<TDim> const&) -> void
        {
            auto const item = sycl::ext::oneapi::experimental::this_nd_item<TDim::value>();
            item.barrier();
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockCount, BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const&, int predicate) -> int
        {
            auto const item = sycl::ext::oneapi::experimental::this_nd_item<TDim::value>();
            item.barrier();

            auto const group = item.get_group();
            auto const counter = (predicate != 0) ? 1 : 0;
            return sycl::reduce_over_group(group, counter, sycl::plus<>{});
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockAnd, BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const&, int predicate) -> int
        {
            auto const item = sycl::ext::oneapi::experimental::this_nd_item<TDim::value>();
            item.barrier();

            auto const group = item.get_group();
            return static_cast<int>(sycl::all_of_group(group, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockOr, BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const&, int predicate) -> int
        {
            auto const item = sycl::ext::oneapi::experimental::this_nd_item<TDim::value>();
            item.barrier();

            auto const group = item.get_group();
            return static_cast<int>(sycl::any_of_group(group, static_cast<bool>(predicate)));
        }
    };
} // namespace alpaka::trait

#endif
