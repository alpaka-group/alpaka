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

#include <alpaka/interfaces/Stream.hpp> // alpaka::event::StreamEnqueueEvent, ...

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

    namespace stream
    {
        namespace detail
        {
            //#############################################################################
            //! Waits for the completion of the given stream.
            //#############################################################################
            template<typename TStream>
            struct ThreadWaitStream<
                TStream, 
                typename std::enable_if<std::is_base_of<host::detail::StreamHost, TStream>::value, void>::type>
            {
                ALPAKA_FCT_HOST ThreadWaitStream(host::detail::StreamHost const &)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };

            //#############################################################################
            //! Waits the stream for the completion of the given event.
            //#############################################################################
            template<typename TStream, typename TEvent>
            struct StreamWaitEvent<
                TStream,
                TEvent,
                typename std::enable_if<std::is_base_of<host::detail::StreamHost, TStream>::value && std::is_same<typename TStream::Acc, typename TEvent::Acc>::value, void>::type>
            {
                ALPAKA_FCT_HOST StreamWaitEvent(host::detail::StreamHost const &, TEvent const &)
                {
                    // Because host calls are not asynchronous, this call never has to let a stream wait.
                }
            };

            //#############################################################################
            //! Tests if all operations in the given stream have been completed.
            //#############################################################################
            template<typename TStream>
            struct StreamTest<
                TStream,
                typename std::enable_if<std::is_base_of<host::detail::StreamHost, TStream>::value, void>::type>
            {
                ALPAKA_FCT_HOST StreamTest(host::detail::StreamHost const &, bool & bTest)
                {
                    // Because host calls are not asynchronous, this call always returns true.
                    bTest = true;
                }
            };
        }
    }
}