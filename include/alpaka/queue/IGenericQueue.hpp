/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

// #include <alpaka/event/EventGeneric.hpp>
// #include <alpaka/dev/DevCpu.hpp>

namespace alpaka
{
    namespace event
    {
        template<typename TDev>
        class EventGeneric;
    }
}

namespace alpaka
{
    namespace queue
    {
#if BOOST_COMP_CLANG
    // avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit [-Werror,-Wweak-vtables]"
    // https://stackoverflow.com/a/29288300
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif

        //#############################################################################
        //! The CPU queue interface
        template<
            typename TDev>
        class IGenericQueue
        {
        public:
            //-----------------------------------------------------------------------------
            //! enqueue the event
            virtual void enqueue(event::EventGeneric<TDev> &) = 0;
            //-----------------------------------------------------------------------------
            //! waiting for the event
            virtual void wait(event::EventGeneric<TDev> const &) = 0;
            //-----------------------------------------------------------------------------
            virtual ~IGenericQueue() = default;
        };
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
    }
}
