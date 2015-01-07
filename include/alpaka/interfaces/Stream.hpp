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

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

// forward declarations
namespace alpaka
{
    namespace event
    {
        template<typename TAcc>
        class Event;
    }
}

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The stream management functionality.
    //-----------------------------------------------------------------------------
    namespace stream
    {
        //#############################################################################
        //! The abstract stream.
        //#############################################################################
        template<typename TAcc>
        class Stream;

        namespace detail
        {
            //#############################################################################
            //! The abstract thread stream waiter.
            //#############################################################################
            template<typename TStream, typename TSfinae = void>
            struct ThreadWaitStream;

            //#############################################################################
            //! The abstract stream event waiter.
            //#############################################################################
            template<typename TStream, typename TEvent, typename TSfinae = void>
            struct StreamWaitEvent;

            //#############################################################################
            //! The abstract thread stream waiter.
            //#############################################################################
            template<typename TStream, typename TSfinae = void>
            struct StreamTest;
        }

        //#############################################################################
        //! Waits for the completion of the given stream.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST void wait(Stream<TAcc> const & stream)
        {
            detail::ThreadWaitStream<Stream<TAcc>>{stream};
        }

        //#############################################################################
        //! Waits the stream for the completion of the given event.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST void wait(Stream<TAcc> const & stream, event::Event<TAcc> const & event)
        {
            detail::StreamWaitEvent<Stream<TAcc>, event::Event<TAcc>>{stream, event};
        }

        //#############################################################################
        //! Tests if all operations in the given stream have been completed.
        //#############################################################################
        template<typename TAcc>
        ALPAKA_FCT_HOST bool test(Stream<TAcc> const & stream)
        {
            bool bTest(false);

            detail::StreamTest<Stream<TAcc>>{stream, bTest};

            return bTest;
        }
    }
}