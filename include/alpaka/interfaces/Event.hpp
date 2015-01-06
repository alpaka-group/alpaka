/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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
* GNU Lesser General Public License
*
*
* You should have received a copy of the GNU Lesser General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The event management functionality.
    //-----------------------------------------------------------------------------
    namespace event
    {
        //#############################################################################
        //! The abstract event.
        //#############################################################################
        template<typename TAcc>
        class Event;

        namespace detail
        {
            //#############################################################################
            //! The abstract event enqueuer.
            //#############################################################################
            template<typename TEvent, typename TSfinae = void>
            struct EventEnqueue;

            //#############################################################################
            //! The abstract event waiter.
            //#############################################################################
            template<typename TEvent, typename TSfinae = void>
            struct EventWait;

            //#############################################################################
            //! The abstract event tester.
            //#############################################################################
            template<typename TEvent, typename TSfinae = void>
            struct EventTest;
        }

        //#############################################################################
        //! Queues the given event.
        //!
        //! If it has previously been queued, then this call will overwrite any existing state of the event. 
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST void enqueue(Event<TAcc> const & event)
        {
            detail::EventEnqueue<Event<TAcc>>{event};
        }

        //#############################################################################
        //! Waits for the completion of the given event.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST void wait(Event<TAcc> const & event)
        {
            detail::EventWait<Event<TAcc>>{event};
        }

        //#############################################################################
        //! Tests if the given event has already be completed.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST bool test(Event<TAcc> const & event)
        {
            bool bTest(false);

            detail::EventTest<Event<TAcc>>{event, bTest};

            return bTest;
        }
    }
}
