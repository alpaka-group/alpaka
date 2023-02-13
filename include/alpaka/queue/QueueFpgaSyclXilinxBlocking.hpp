/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#    include <alpaka/queue/QueueGenericSyclBlocking.hpp>

namespace alpaka::experimental
{
    using QueueFpgaSyclXilinxBlocking = QueueGenericSyclBlocking<DevFpgaSyclXilinx>;
}

#endif
