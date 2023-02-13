/* Copyright 2020 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/dev/DevOmp5.hpp>
#    include <alpaka/event/EventGenericThreads.hpp>

namespace alpaka
{
    using EventOmp5 = EventGenericThreads<DevOmp5>;
}

#endif
