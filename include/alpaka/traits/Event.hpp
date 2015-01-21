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
            //#############################################################################
            template<
                typename TAcc>
            class GetEvent;

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
        //#############################################################################
        template<
            typename TAcc>
        using GetEventT = typename traits::event::GetEvent<TAcc>::type;

        //-----------------------------------------------------------------------------
        //! Queues the given event in the stream zero.
        //!
        //! If it has previously been queued, then this call will overwrite any existing state of the event. 
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        //-----------------------------------------------------------------------------
        template<
            typename TEvent>
        ALPAKA_FCT_HOST void enqueue(
            TEvent const & event)
        {
            traits::event::DefaultStreamEnqueueEvent<TEvent>::defaultStreamEnqueueEvent(event);
        }

        //-----------------------------------------------------------------------------
        //! Queues the given event in the given stream.
        //!
        //! If it has previously been queued, then this call will overwrite any existing state of the event. 
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        //-----------------------------------------------------------------------------
        template<
            typename TEvent,
            typename TStream>
        ALPAKA_FCT_HOST void enqueue(
            TEvent const & event,
            TStream const & stream)
        {
            traits::event::StreamEnqueueEvent<TEvent, TStream>::streamEnqueueEvent(event, stream);
        }

        //-----------------------------------------------------------------------------
        //! Tests if the given event has already be completed.
        //-----------------------------------------------------------------------------
        template<
            typename TEvent>
        ALPAKA_FCT_HOST bool test(
            TEvent const & event)
        {
            return traits::event::EventTest<TEvent>::eventTest(event);
        }
    }
}
