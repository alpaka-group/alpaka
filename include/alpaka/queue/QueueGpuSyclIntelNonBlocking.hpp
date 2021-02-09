/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI)

#include <alpaka/dev/DevGpuSyclIntel.hpp>
#include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>

namespace alpaka
{
    using QueueGpuSyclIntelNonBlocking = QueueGenericSyclNonBlocking<DevGpuSyclIntel>;
}

#endif
