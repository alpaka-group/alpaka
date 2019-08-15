/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <memory>

namespace alpaka
{
    namespace event
    {
        class EventCpu;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace cpu
        {

            //#############################################################################
            //! The CPU queue interface
            class ICpuQueue
            {
            public:
                //-----------------------------------------------------------------------------
                ICpuQueue() = default;
                //! enqueue the event into the given queue ------------------------------------
                virtual void enqueue(std::shared_ptr<ICpuQueue> &, event::EventCpu &) = 0;
                //! the given queue is waiting for the event-----------------------------------
                virtual void wait(std::shared_ptr<ICpuQueue> &, event::EventCpu const &) = 0;
                //-----------------------------------------------------------------------------
                virtual ~ICpuQueue() = default;
            };
        }
    }
}
