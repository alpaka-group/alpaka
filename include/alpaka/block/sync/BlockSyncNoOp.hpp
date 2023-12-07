/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Axel Hübl <a.huebl@plasma.ninja>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 * SPDX-FileContributor: Matthias Werner <Matthias.Werner1@tu-dresden.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/sync/Traits.hpp"
#include "alpaka/core/Common.hpp"

namespace alpaka
{
    //! The no op block synchronization.
    class BlockSyncNoOp : public concepts::Implements<ConceptBlockSync, BlockSyncNoOp>
    {
    };

    namespace trait
    {
        template<>
        struct SyncBlockThreads<BlockSyncNoOp>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreads(BlockSyncNoOp const& /* blockSync */) -> void
            {
                // Nothing to do.
            }
        };

        template<typename TOp>
        struct SyncBlockThreadsPredicate<TOp, BlockSyncNoOp>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(BlockSyncNoOp const& /* blockSync */, int predicate)
                -> int
            {
                return predicate;
            }
        };
    } // namespace trait
} // namespace alpaka
