/**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Benjamin Worpitz
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

#include <alpaka/traits/Wait.hpp>   // CurrentThreadWaitFor, WaiterWaitFor

namespace alpaka
{
    namespace traits
    {
        //-----------------------------------------------------------------------------
        //! The stream traits.
        //-----------------------------------------------------------------------------
        namespace stream
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
            //! The stream get trait.
            //#############################################################################
            template<
                typename T, 
                typename TSfinae = void>
            struct GetStream;
        }
    }

    //-----------------------------------------------------------------------------
    //! The stream trait accessors.
    //-----------------------------------------------------------------------------
    namespace stream
    {
        //#############################################################################
        //! The stream type trait alias template to remove the ::type.
        //#############################################################################
        template<
            typename TAcc>
        using StreamT = typename traits::stream::StreamType<TAcc>::type;

        //-----------------------------------------------------------------------------
        //! Tests if all ops in the given stream have been completed.
        //-----------------------------------------------------------------------------
        template<
            typename TStream>
        ALPAKA_FCT_HOST bool test(
            TStream const & stream)
        {
            return traits::stream::StreamTest<
                TStream>
            ::streamTest(
                stream);
        }

        //-----------------------------------------------------------------------------
        //! \return The stream.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_HOST_ACC auto getStream(
            T const & type)
        -> decltype(traits::stream::GetStream<T>::getStream(std::declval<T const &>()))
        {
            return traits::stream::GetStream<
                T>
            ::getStream(
                type);
        }
    }
}
