/* Copyright 2024 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

namespace alpaka
{
    struct ConceptGridSync
    {
    };

    //! The grid synchronization traits.
    namespace trait
    {
        //! The grid synchronization operation trait.
        template<typename TGridSync, typename TSfinae = void>
        struct SyncGridThreads;

    } // namespace trait

    // TODO: investigate lock-ups with cuda clang

    //! Synchronizes all threads within the current grid. Works only for cooperative kernels.
    //! NOTE: when compiled with CUDA Clang locks up if numberOfBlocks > 2 * multiProcessorCount.
    //! Consider switching to nvcc.
    //!
    //! \tparam TGridSync The grid synchronization implementation type.
    //! \param gridSync The grid synchronization implementation.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TGridSync>
    ALPAKA_FN_ACC auto syncGridThreads(TGridSync const& gridSync) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptGridSync, TGridSync>;
        trait::SyncGridThreads<ImplementationBase>::syncGridThreads(gridSync);
    }


} // namespace alpaka
