/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/ConcurrentExecPool.hpp>

#include <type_traits>
#include <thread>
#include <mutex>
#include <future>

namespace alpaka
{
    namespace queue
    {
        using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<dev::DevCpu>;
    }
}

#include <alpaka/event/EventCpu.hpp>
