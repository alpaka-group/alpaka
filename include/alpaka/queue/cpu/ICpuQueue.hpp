/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jeffrey Kelling <j.kelling@hzdr.de>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/queue/cpu/IGenericThreadsQueue.hpp"

namespace alpaka::cpu
{
    //! The CPU queue interface
    using ICpuQueue = IGenericThreadsQueue<DevCpu>;
} // namespace alpaka::cpu
