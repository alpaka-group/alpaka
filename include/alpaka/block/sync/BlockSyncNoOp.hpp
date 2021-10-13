/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/sync/Traits.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

namespace alpaka
{
    //! The no op block synchronization.
    class BlockSyncNoOp : public concepts::Implements<ConceptBlockSync, BlockSyncNoOp>
    {
    };

    namespace traits
    {
        template<>
        struct SyncBlockThreads<BlockSyncNoOp>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreads(BlockSyncNoOp const& blockSync) -> void
            {
                alpaka::ignore_unused(blockSync);
                // Nothing to do.
            }
        };

        template<typename TOp>
        struct SyncBlockThreadsPredicate<TOp, BlockSyncNoOp>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(BlockSyncNoOp const& blockSync, int predicate) -> int
            {
                alpaka::ignore_unused(blockSync);
                return predicate;
            }
        };
    } // namespace traits
} // namespace alpaka
