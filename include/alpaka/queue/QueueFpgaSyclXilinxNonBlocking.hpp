/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevFpgaSyclXilinx.hpp"
#include "alpaka/queue/QueueGenericSyclNonBlocking.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

namespace alpaka
{
    using QueueFpgaSyclXilinxNonBlocking = QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>;
}

#endif
