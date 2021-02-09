/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/block/sync/Traits.hpp>

#include <CL/sycl.hpp>

namespace alpaka
{
    //#############################################################################
    //! The SYCL block synchronization.
    template <typename TDim>
    class BlockSyncGenericSycl : public concepts::Implements<ConceptBlockSync, BlockSyncGenericSycl<TDim>>
    {
    public:
        using BlockSyncBase = BlockSyncGenericSycl<TDim>;

        //-----------------------------------------------------------------------------
        BlockSyncGenericSycl(cl::sycl::nd_item<TDim::value> work_item)
        : my_item{work_item}
        {
        }
        //-----------------------------------------------------------------------------
        BlockSyncGenericSycl(BlockSyncGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        BlockSyncGenericSycl(BlockSyncGenericSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSyncGenericSycl const &) -> BlockSyncGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSyncGenericSycl &&) -> BlockSyncGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSyncGenericSycl() = default;

        cl::sycl::nd_item<TDim::value> my_item;
    };

    namespace traits
    {
        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreads<BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreads(BlockSyncGenericSycl<TDim> const & blockSync) -> void
            {
                blockSync.my_item.barrier();
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockCount, BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const & blockSync, int predicate) -> int
            {
                using namespace cl::sycl;

                const auto group = blockSync.my_item.get_group();
                blockSync.my_item.barrier();

                const auto counter = (predicate != 0) ? 1 : 0;
                return ONEAPI::reduce(group, counter, ONEAPI::plus<>{});
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockAnd, BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const & blockSync, int predicate) -> int
            {
                using namespace cl::sycl;

                const auto group = blockSync.my_item.get_group();
                blockSync.my_item.barrier();

                return static_cast<int>(ONEAPI::all_of(group, static_cast<bool>(predicate)));
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockOr, BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const & blockSync, int predicate) -> int
            {
                using namespace cl::sycl;

                const auto group = blockSync.my_item.get_group();
                blockSync.my_item.barrier();

                return static_cast<int>(ONEAPI::any_of(group, static_cast<bool>(predicate)));
            }
        };
    }
}

#endif
