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

#include <alpaka/wait/Traits.hpp>   // CurrentThreadWaitFor, WaiterWaitFor

#include <alpaka/core/Common.hpp>   // ALPAKA_FN_HOST

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The stream specifics.
    //-----------------------------------------------------------------------------
    namespace stream
    {
        //-----------------------------------------------------------------------------
        //! The stream traits.
        //-----------------------------------------------------------------------------
        namespace traits
        {
            //#############################################################################
            //! The stream type trait.
            //#############################################################################
            template<
                typename TAcc,
                typename TSfinae = void>
            struct StreamType;

            //#############################################################################
            //! The stream test trait.
            //#############################################################################
            template<
                typename TStream,
                typename TSfinae = void>
            struct StreamTest;

            //#############################################################################
            //! The stream enqueue trait.
            //#############################################################################
            template<
                typename TStream,
                typename TEvent,
                typename TSfinae = void>
            struct StreamEnqueue;

            //#############################################################################
            //! The stream get trait.
            //#############################################################################
            template<
                typename T,
                typename TSfinae = void>
            struct GetStream;
        }

        //#############################################################################
        //! The stream type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TAcc>
        using StreamT = typename traits::StreamType<TAcc>::type;

        //-----------------------------------------------------------------------------
        //! Creates a stream on a device.
        //-----------------------------------------------------------------------------
        template<
            typename TDev>
        ALPAKA_FN_HOST auto create(
            TDev & dev)
        -> StreamT<TDev>
        {
            return StreamT<TDev>(dev);
        }

        //-----------------------------------------------------------------------------
        //! Queues the given task in the given stream.
        //!
        //! If it has previously been queued, then this call will overwrite any existing state of the event.
        //! Any subsequent calls which examine the status of event will only examine the completion of this most recent call to enqueue.
        //-----------------------------------------------------------------------------
        template<
            typename TStream,
            typename TTask>
        ALPAKA_FN_HOST auto enqueue(
            TStream & stream,
            TTask & task)
        -> void
        {
            traits::StreamEnqueue<
                TStream,
                TTask>
            ::streamEnqueue(
                stream,
                task);
        }

        //-----------------------------------------------------------------------------
        //! Tests if all ops in the given stream have been completed.
        //-----------------------------------------------------------------------------
        template<
            typename TStream>
        ALPAKA_FN_HOST auto test(
            TStream const & stream)
        -> bool
        {
            return traits::StreamTest<
                TStream>
            ::streamTest(
                stream);
        }

        //-----------------------------------------------------------------------------
        //! \return The stream.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FN_HOST_ACC auto getStream(
            T const & type)
        -> decltype(traits::GetStream<T>::getStream(type))
        {
            return traits::GetStream<
                T>
            ::getStream(
                type);
        }
    }
}
