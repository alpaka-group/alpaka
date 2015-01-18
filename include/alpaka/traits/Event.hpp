/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>   // ALPAKA_FCT_HOST

// forward declarations
namespace alpaka
{
    namespace stream
    {
        template<
            typename TAcc>
        class Stream;
    }
}

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The event management traits.
        //-----------------------------------------------------------------------------
        namespace event
        {
            //#############################################################################
            //! The event type trait.
            // \TODO: Implement GetEventT/GetEvent!
            //#############################################################################
            //template<
            //    typename TAcc>
            //class GetEvent;

            //#############################################################################
            //! The event enqueuer trait.
            //#############################################################################
            template<
                typename TEvent, 
                typename TSfinae = void>
            struct DefaultStreamEnqueueEvent;

            //#############################################################################
            //! The event enqueuer trait.
            //#############################################################################
            template<
                typename TEvent, 
                typename TStream, 
                typename TSfinae = void>
            struct StreamEnqueueEvent;

            //#############################################################################
            //! The thread event waiter trait.
            //#############################################################################
            template<
                typename TEvent, 
                typename TSfinae = void>
            struct ThreadWaitEvent;

            //#############################################################################
            //! The event tester trait.
            //#############################################################################
            template<
                typename TEvent, 
                typename TSfinae = void>
            struct EventTest;
        }
    }

    //-----------------------------------------------------------------------------
    //! The event management trait accessors.
    //-----------------------------------------------------------------------------
    namespace event
    {
        //#############################################################################
        //! The event type trait alias template to remove the ::type.
        // \TODO: Implement GetEventT/GetEvent!
        //#############################################################################
        //template<
        //    typename TAcc> 
        //using GetEventT = typename traits::memory::GetEvent<TAcc>::type;

        //#############################################################################
        //! The abstract event.
        //#############################################################################
        template<
            typename TAcc> 
        class Event;

        //-----------------------------------------------------------------------------
        //! Queues the given event in the stream zero.
        //!
        //! If it has previously been queued, then this call will overwrite any existing state of the event. 
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc>
        ALPAKA_FCT_HOST void enqueue(
            Event<TAcc> const & event)
        {
            traits::event::DefaultStreamEnqueueEvent<Event<TAcc>>::defaultStreamEnqueueEvent(event);
        }

        //-----------------------------------------------------------------------------
        //! Queues the given event in the given stream.
        //!
        //! If it has previously been queued, then this call will overwrite any existing state of the event. 
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc>
        ALPAKA_FCT_HOST void enqueue(
            Event<TAcc> const & event, 
            stream::Stream<TAcc> const & stream)
        {
            traits::event::StreamEnqueueEvent<Event<TAcc>, stream::Stream<TAcc>>::streamEnqueueEvent(event, stream);
        }

        //-----------------------------------------------------------------------------
        //! Waits the thread for the completion of the given event.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc>
        ALPAKA_FCT_HOST void wait(
            Event<TAcc> const & event)
        {
            traits::event::ThreadWaitEvent<Event<TAcc>>::threadWaitEvent(event);
        }

        //-----------------------------------------------------------------------------
        //! Tests if the given event has already be completed.
        //-----------------------------------------------------------------------------
        template<
            typename TAcc>
        ALPAKA_FCT_HOST bool test(
            Event<TAcc> const & event)
        {
            return traits::event::EventTest<Event<TAcc>>::eventTest(event);
        }
    }
}
