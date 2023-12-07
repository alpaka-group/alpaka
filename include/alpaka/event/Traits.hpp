/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dev/Traits.hpp"

namespace alpaka
{
    //! The event management traits.
    namespace trait
    {
        //! The event type trait.
        template<typename T, typename TSfinae = void>
        struct EventType;

        //! The event tester trait.
        template<typename TEvent, typename TSfinae = void>
        struct IsComplete;
    } // namespace trait

    //! The event type trait alias template to remove the ::type.
    template<typename T>
    using Event = typename trait::EventType<T>::type;

    //! Tests if the given event has already been completed.
    //!
    //! \warning This function is allowed to return false negatives. An already completed event can reported as
    //! uncompleted because the status information are not fully propagated by the used alpaka backend.
    //! \return true event is finished/complete else false.
    template<typename TEvent>
    ALPAKA_FN_HOST auto isComplete(TEvent const& event) -> bool
    {
        return trait::IsComplete<TEvent>::isComplete(event);
    }
} // namespace alpaka
