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
            class StreamHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamHost() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamHost(StreamHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamHost(StreamHost &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST StreamHost & operator=(StreamHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Equality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator==(StreamHost const &) const
                {
                    return true;
                }
                //-----------------------------------------------------------------------------
                //! Inequality comparison operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST bool operator!=(StreamHost const & rhs) const
                {
                    return !((*this) == rhs);
                }
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~StreamHost() noexcept = default;
            };
        }
    }

    namespace traits
    {
        namespace stream
        {
            //#############################################################################
            //! Tests if all operations in the given stream have been completed.
            //#############################################################################
            template<
                typename TStream>
            struct StreamTest<
                TStream,
                typename std::enable_if<std::is_base_of<host::detail::StreamHost, TStream>::value>::type>
            {
                static ALPAKA_FCT_HOST bool streamTest(
                    TStream const &)
                {
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
                typename TStream>
            struct CurrentThreadWaitFor<
                TStream,
                typename std::enable_if<std::is_base_of<host::detail::StreamHost, TStream>::value>::type>
            {
                ALPAKA_FCT_HOST static void currentThreadWaitFor(
                    TStream const & stream)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };

            //#############################################################################
            //! The CUDA accelerator stream event wait trait specialization.
            //#############################################################################
            template<
                typename TStream,
                typename TEvent>
            struct WaiterWaitFor<
                TStream,
                TEvent,
                typename std::enable_if<
                    std::is_base_of<host::detail::StreamHost, TStream>::value
                    && std::is_same<typename alpaka::acc::GetAccT<TStream>, typename alpaka::acc::GetAccT<TEvent>>::value>::type>
            {
                ALPAKA_FCT_HOST static void waiterWaitFor(
                    TStream const & stream,
                    TEvent const & event)
                {
                    // Because host calls are not asynchronous, this call never has to let a stream wait.
                }
            };
        }
    }
}