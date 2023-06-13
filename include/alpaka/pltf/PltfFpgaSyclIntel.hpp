/* Copyright 2023 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/pltf/PltfGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

#    include <CL/sycl.hpp>
#    include <sycl/ext/intel/fpga_extensions.hpp>

namespace alpaka
{
    //! The SYCL device manager.
#    ifdef ALPAKA_FPGA_EMULATION
    using PltfFpgaSyclIntel = PltfGenericSycl<sycl::ext::intel::fpga_emulator_selector>;
#    else
    using PltfFpgaSyclIntel = PltfGenericSycl<sycl::ext::intel::fpga_selector>;
#    endif
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<PltfFpgaSyclIntel>
    {
        using type = DevGenericSycl<PltfFpgaSyclIntel>; // = DevFpgaSyclIntel
    };
} // namespace alpaka::trait

#endif
