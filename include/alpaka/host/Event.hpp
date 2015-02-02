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

#include <alpaka/traits/Event.hpp>      // StreamEnqueueEvent, ...
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor

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
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost(EventHost &&) = default;
#endif
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

    namespace traits
    {
        namespace event
        {
            //#############################################################################
            //! The host accelerators event enqueue trait specialization.
            //#############################################################################
            template<
                typename TEvent>
            struct DefaultStreamEnqueueEvent<
                TEvent,
                typename std::enable_if<std::is_base_of<host::detail::EventHost, TEvent>::value>::type>
            {
                ALPAKA_FCT_HOST static void defaultStreamEnqueueEvent(
                    host::detail::EventHost const & event)
                {
                    boost::ignore_unused(event);
                    // Because host calls are not asynchronous, this call never has to enqueue anything.
                }
            };

            //#############################################################################
            //! The host accelerators event enqueue trait specialization.
            //#############################################################################
            template<
                typename TEvent, 
                typename TStream>
            struct StreamEnqueueEvent<
                TEvent,
                TStream,
                typename std::enable_if<
                    std::is_base_of<host::detail::EventHost, TEvent>::value 
                    && std::is_same<typename alpaka::acc::GetAccT<TEvent>, typename alpaka::acc::GetAccT<TStream>>::value>::type>
            {
                ALPAKA_FCT_HOST static void streamEnqueueEvent(
                    host::detail::EventHost const & event, 
                    TStream const & stream)
                {
                    boost::ignore_unused(event);
                    boost::ignore_unused(stream);
                    // Because host calls are not asynchronous, this call never has to enqueue anything.
                }
            };

            //#############################################################################
            //! The host accelerators event test trait specialization.
            //#############################################################################
            template<
                typename TEvent>
            struct EventTest<
                TEvent,
                typename std::enable_if<std::is_base_of<host::detail::EventHost, TEvent>::value>::type>
            {
                ALPAKA_FCT_HOST static bool eventTest(
                    host::detail::EventHost const & event)
                {
                    boost::ignore_unused(event);
                    // Because host calls are not asynchronous, this call always returns true.
                    return true;
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The host accelerator event thread wait trait specialization.
            //#############################################################################
            template<
                typename TEvent>
            struct CurrentThreadWaitFor<
                TEvent,
                typename std::enable_if<std::is_base_of<host::detail::EventHost, TEvent>::value>::type>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    TEvent const & event)
                {
                    boost::ignore_unused(event);
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}