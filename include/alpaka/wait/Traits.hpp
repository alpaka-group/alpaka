/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

namespace alpaka
{
    struct ConceptCurrentThreadWaitFor
    {
    };

    //! The wait traits.
    namespace trait
    {
        //! The thread wait trait.
        template<typename TAwaited, typename TSfinae = void>
        struct CurrentThreadWaitFor;

        //! The waiter wait trait.
        template<typename TWaiter, typename TAwaited, typename TSfinae = void>
        struct WaiterWaitFor;
    } // namespace trait

    //! Waits the thread for the completion of the given awaited action to complete.
    template<typename TAwaited>
    ALPAKA_FN_HOST auto wait(TAwaited const& awaited) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptCurrentThreadWaitFor, TAwaited>;
        trait::CurrentThreadWaitFor<ImplementationBase>::currentThreadWaitFor(awaited);
    }

    //! The waiter waits for the given awaited action to complete.
    template<typename TWaiter, typename TAwaited>
    ALPAKA_FN_HOST auto wait(TWaiter& waiter, TAwaited const& awaited) -> void
    {
        trait::WaiterWaitFor<TWaiter, TAwaited>::waiterWaitFor(waiter, awaited);
    }
} // namespace alpaka
