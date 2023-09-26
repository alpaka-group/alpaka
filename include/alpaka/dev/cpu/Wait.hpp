/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 * SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 * SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/event/EventCpu.hpp"
#include "alpaka/wait/Traits.hpp"

namespace alpaka::trait
{
    //! The CPU device thread wait specialization.
    //!
    //! Blocks until the device has completed all preceding requested tasks.
    //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
    template<>
    struct CurrentThreadWaitFor<DevCpu>
    {
        ALPAKA_FN_HOST static auto currentThreadWaitFor(DevCpu const& dev) -> void
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            generic::currentThreadWaitForDevice(dev);
        }
    };
} // namespace alpaka::trait
