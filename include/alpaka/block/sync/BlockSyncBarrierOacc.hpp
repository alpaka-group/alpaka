/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENACC

#include <alpaka/block/sync/Traits.hpp>

namespace alpaka
{
    //#############################################################################
    //! The OpenACC barrier block synchronization.
    //! Traits are specialized on BlockSyncBarrierOaccBlockShared
    class BlockSyncBarrierOacc
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierOacc()
        {}
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierOacc(BlockSyncBarrierOacc const &) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST BlockSyncBarrierOacc(BlockSyncBarrierOacc &&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOacc const &) -> BlockSyncBarrierOacc & = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOacc &&) -> BlockSyncBarrierOacc & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSyncBarrierOacc() = default;

        class BlockShared : public concepts::Implements<ConceptBlockSync, BlockShared>
        {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockShared()
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockShared(BlockShared const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockShared(BlockShared &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockShared const &) -> BlockShared & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockShared &&) -> BlockShared & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockShared() = default;

                std::uint8_t mutable m_generation = 0u;
                // NVHPC 20.7: initializer causes warning:
                // NVC++-W-0155-External and Static variables are not supported in acc routine - _T139951818207704_37530
                int mutable m_syncCounter[4] {0,0,0,0};
                int mutable m_result;
        };
    };
}

#endif
