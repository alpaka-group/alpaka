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

#include <alpaka/traits/Stream.hpp>     // traits::StreamEnqueueEvent, ...
#include <alpaka/traits/Wait.hpp>       // CurrentThreadWaitFor, WaiterWaitFor
#include <alpaka/traits/Acc.hpp>        // AccT
#include <alpaka/traits/Dev.hpp>        // GetDev

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::is_base

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The host accelerators stream.
            //#############################################################################
            template<
                typename TDev>
            class StreamHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamHost(
                    TDev const & dev) :
                        m_Dev(dev)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamHost(StreamHost const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamHost(StreamHost &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(StreamHost const &) -> StreamHost & = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator==(StreamHost const & rhs) const
                -> bool
                {
                    return (m_Dev == rhs.m_Dev);
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator!=(StreamHost const & rhs) const
                -> bool
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~StreamHost() noexcept = default;

            public:
                TDev m_Dev;
            };

            template<
                typename TDev>
            class EventHost;
        }
    }

    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The host accelerators stream device get trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct GetDev<
                host::detail::StreamHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    host::detail::StreamHost<TDev> const & stream)
                -> TDev
                {
                    return stream.m_Dev;
                }
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The host accelerators stream stream type trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct StreamType<
                host::detail::StreamHost<TDev>>
            {
                using type = host::detail::StreamHost<TDev>;
            };

            //#############################################################################
            //! The host accelerators stream test trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct StreamTest<
                host::detail::StreamHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto streamTest(
                    host::detail::StreamHost<TDev> const & stream)
                -> bool
                {
                    boost::ignore_unused(stream);
                    // Because host calls are not asynchronous, this call always returns true.
                    return true;
                }
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The host accelerators stream thread wait trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct CurrentThreadWaitFor<
                host::detail::StreamHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    host::detail::StreamHost<TDev> const & stream)
                -> void
                {
                    boost::ignore_unused(stream);
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };

            //#############################################################################
            //! The host accelerators stream event wait trait specialization.
            //#############################################################################
            template<
                typename TDev>
            struct WaiterWaitFor<
                host::detail::StreamHost<TDev>,
                host::detail::EventHost<TDev>>
            {
                ALPAKA_FCT_HOST static auto waiterWaitFor(
                    host::detail::StreamHost<TDev> const & stream,
                    host::detail::EventHost<TDev> const & event)
                -> void
                {
                    boost::ignore_unused(stream);
                    boost::ignore_unused(event);
                    // Because host calls are not asynchronous, this call never has to let a stream wait.
                }
            };
        }
    }
}