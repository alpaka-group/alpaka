/* Copyright 2019 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/dev/DevOmp4.hpp>
#include <alpaka/event/EventGenericThreads.hpp>

namespace alpaka
{
    namespace event
    {
        using EventOmp4 = EventGenericThreads<dev::DevOmp4>;
    }
}

#endif
