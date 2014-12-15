/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/interfaces/Event.hpp>

namespace alpaka
{
    namespace host
    {
        namespace detail
        {
            //#############################################################################
            //! The template for an event.
            //#############################################################################
            class EventHost
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost(EventHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost(EventHost &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST EventHost & operator=(EventHost const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~EventHost() noexcept = default;
            };
        }
    }

    namespace event
    {
        namespace detail
        {
            //#############################################################################
            //! The template for enqueuing the given event.
            //#############################################################################
            template<typename TAcc>
            struct EventEnqueue
            {
                ALPAKA_FCT_HOST EventEnqueue(host::detail::EventHost const & )
                {
                    // Because host calls are not asynchronous, this call never has to enqueue anything.
                }
            };

            //#############################################################################
            //! The template for an event wait.
            //#############################################################################
            template<typename TAcc>
            struct EventWait
            {
                ALPAKA_FCT_HOST EventWait(host::detail::EventHost const & )
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                }
            };

            //#############################################################################
            //! The template for an event test.
            //#############################################################################
            template<typename TAcc>
            struct EventTest
            {
                ALPAKA_FCT_HOST EventTest(host::detail::EventHost const & , bool & bTest)
                {
                    // Because host calls are not asynchronous, this call never has to wait.
                    bTest = true;
                }
            };
        }
    }
}