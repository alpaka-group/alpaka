/* Copyright 2020 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

#    include <alpaka/dev/DevOacc.hpp>
#    include <alpaka/queue/QueueGenericThreadsBlocking.hpp>

namespace alpaka
{
    using QueueOaccBlocking = QueueGenericThreadsBlocking<DevOacc>;
}

#endif
