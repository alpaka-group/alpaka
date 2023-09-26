/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Felice Pantaleo <felice.pantaleo@cern.ch>
 * SPDX-FileContributor: René Widera <r.widera@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <boost/version.hpp>

#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#    define ALPAKA_DISABLE_ATOMIC_ATOMICREF
#endif

#include "alpaka/atomic/AtomicAtomicRef.hpp"
#include "alpaka/atomic/AtomicStdLibLock.hpp"

namespace alpaka
{
#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
    using AtomicCpu = AtomicAtomicRef;
#else
    using AtomicCpu = AtomicStdLibLock<16>;
#endif // ALPAKA_DISABLE_ATOMIC_ATOMICREF

} // namespace alpaka
