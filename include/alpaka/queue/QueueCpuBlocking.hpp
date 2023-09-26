/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/event/EventCpu.hpp"
#include "alpaka/queue/QueueGenericThreadsBlocking.hpp"

namespace alpaka
{
    using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;
} // namespace alpaka
