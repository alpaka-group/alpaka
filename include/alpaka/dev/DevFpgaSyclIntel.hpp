/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/pltf/PltfFpgaSyclIntel.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

namespace alpaka
{
    using DevFpgaSyclIntel = DevGenericSycl<PltfFpgaSyclIntel>;
} // namespace alpaka

#endif
