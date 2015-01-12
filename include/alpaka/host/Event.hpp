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

#include <type_traits>                  // std::is_base

#include <alpaka/interfaces/Event.hpp>  // alpaka::event::StreamEnqueueEvent, ...

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The host accelerators event.
            //#############################################################################
            class EventHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost(EventHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost(EventHost &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost & operator=(EventHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~EventHost() noexcept = default;
            };
        }
    }

    namespace event
    {
        namespace detail
        {
            //#############################################################################
            //! The host accelerators event enqueuer.
            //#############################################################################
            template<typename TEvent>
            struct DefaultStreamEnqueueEvent<
                TEvent,
                typename std::enable_if<std::is_base_of<host::detail::EventHost, TEvent>::value, void>::type>
            {
                ALPAKA_FCT_HOST DefaultStreamEnqueueEvent(host::detail::EventHost const &)
                {
                    // Because host calls are not asynchronous, this call never has to enqueue anything.
                }
            };

            //#############################################################################
            //! The host accelerators event enqueuer.
            //#############################################################################
            template<typename TEvent, typename TStream>
            struct StreamEnqueueEvent<
                TEvent,
                TStream,
                typename std::enable_if<std::is_base_of<host::detail::EventHost, TEvent>::value && std::is_same<typename TEvent::Acc, typename TStream::Acc>::value, void>::type>
            {
                ALPAKA_FCT_HOST StreamEnqueueEvent(host::detail::EventHost const &, TStream const &)
                {
                    // Because host calls are not asynchronous, this call never has to enqueue anything.
                }
            };

            //#############################################################################
            //! The host accelerators thread event waiter.
            //#############################################################################
            template<typename TEvent>
            struct ThreadWaitEvent<
                TEvent, 
                typename std::enable_if<std::is_base_of<host::detail::EventHost, TEvent>::value, void>::type>
            {
                ALPAKA_FCT_HOST ThreadWaitEvent(host::detail::EventHost const &)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };

            //#############################################################################
            //! The host accelerators event tester.
            //#############################################################################
            template<typename TEvent>
            struct EventTest<
                TEvent,
                typename std::enable_if<std::is_base_of<host::detail::EventHost, TEvent>::value, void>::type>
            {
                ALPAKA_FCT_HOST EventTest(host::detail::EventHost const &, bool & bTest)
                {
                    // Because host calls are not asynchronous, this call always returns true.
                    bTest = true;
                }
            };
        }
    }
}