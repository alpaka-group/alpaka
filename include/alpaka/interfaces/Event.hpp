/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST

namespace alpaka
{
    namespace event
    {
        //#############################################################################
        //! The template for an event.
        //#############################################################################
        template<typename TAcc>
        class Event;

        namespace detail
        {
            //#############################################################################
            //! The template for enqueuing the given event.
            //#############################################################################
            template<typename TAcc>
            struct EventEnqueue;

            //#############################################################################
            //! The template for an event wait.
            //#############################################################################
            template<typename TAcc>
            struct EventWait;

            //#############################################################################
            //! The template for an event test.
            //#############################################################################
            template<typename TAcc>
            struct EventTest;
        }

        //#############################################################################
        //! Queues the given event.
        //! If it has previously been queued, then this call will overwrite any existing state of the event. 
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to eventEnqueue.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST void eventEnqueue(Event<TAcc> const & event)
        {
            detail::EventEnqueue<TAcc>{event};
        }

        //#############################################################################
        //! Waits for the completion of the given event.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST void eventWait(Event<TAcc> const & event)
        {
            detail::EventWait<TAcc>{event};
        }

        //#############################################################################
        //! Tests if the given event has already be completed.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST bool eventTest(Event<TAcc> const & event)
        {
            bool bTest(false);

            detail::EventTest<TAcc>{event, bTest};

            return bTest;
        }
    }
}
