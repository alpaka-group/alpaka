/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevFpgaSyclXilinx.hpp"
#include "alpaka/queue/QueueGenericSyclBlocking.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

namespace alpaka
{
    using QueueFpgaSyclXilinxBlocking = QueueGenericSyclBlocking<DevFpgaSyclXilinx>;
}

#endif
