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

#include <alpaka/threads/AccThreadsFwd.hpp> // AccThreads

#include <alpaka/host/Stream.hpp>           // StreamHost

namespace alpaka
{
    namespace threads
    {
        namespace detail
        {
            //#############################################################################
            //! The threads accelerator stream.
            //#############################################################################
            class StreamThreads :
                public host::detail::StreamHost
            {};
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The threads accelerator stream accelerator type trait specialization.
            //#############################################################################
            template<>
            struct GetAcc<
                threads::detail::StreamThreads>
            {
                using type = AccThreads;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The threads accelerator stream type trait specialization.
            //#############################################################################
            template<>
            class GetStream<
                AccThreads>
            {
                using type = threads::detail::StreamThreads;
            };
        }
    }
}
