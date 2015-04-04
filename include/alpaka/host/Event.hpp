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
#include <alpaka/traits/Dev.hpp>        // GetDev

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The host accelerators event.
            //#############################################################################
            template<
                typename TDev>
            class EventHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost(
                    TDev const & dev) :
                        m_Dev(dev)
                {}
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
                ALPAKA_FCT_HOST auto operator=(EventHost const &) -> EventHost & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(EventHost const & rhs) const
                -> bool
                {
                    return (m_Dev == rhs.m_Dev);
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(EventHost const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~EventHost() noexcept = default;

            public:
                TDev m_Dev;
            };

            template<
                typename TDev>
            class StreamHost;
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The host accelerators event device get trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct GetDev<
                host::detail::EventHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    host::detail::EventHost<TDev> const & event)
                -> TDev
                {
                    return event.m_Dev;
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The host accelerators event event type trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct EventType<
                host::detail::EventHost<TDev>>
            {
                using type = host::detail::EventHost<TDev>;
            };

            //#############################################################################
            //! The host accelerators event enqueue trait specialization.
            //#############################################################################
            /*template<
                typename TDev>
            struct DefaultStreamEnqueueEvent<
                host::detail::EventHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto defaultStreamEnqueueEvent(
                    host::detail::EventHost<TDev> const & event)
                -> void
                {
                    boost::ignore_unused(event);
                    // Because host calls are not asynchronous, this call never has to enqueue anything.
                }
            };*/

            //#############################################################################
            //! The host accelerators event enqueue trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct StreamEnqueueEvent<
                host::detail::EventHost<TDev>,
                host::detail::StreamHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto streamEnqueueEvent(
                    host::detail::EventHost<TDev> const & event,
                    host::detail::StreamHost<TDev> const & stream)
                -> void
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
                typename TDev>
            struct EventTest<
                host::detail::EventHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto eventTest(
                    host::detail::EventHost<TDev> const & event)
                -> bool
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
                typename TDev>
            struct CurrentThreadWaitFor<
                host::detail::EventHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    host::detail::EventHost<TDev> const & event)
                -> void
                {
                    boost::ignore_unused(event);
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };
        }
    }
}