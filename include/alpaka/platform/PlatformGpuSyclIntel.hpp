/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/platform/PlatformGenericSycl.hpp"

#include <string>

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)

#    include <sycl/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        template<>
        struct SYCLDeviceSelector<TagGpuSyclIntel>
        {
            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& vendor = dev.get_info<sycl::info::device::vendor>();
                auto const is_intel_gpu = dev.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos);

                return is_intel_gpu ? 1 : -1;
            }

            static constexpr char name[] = "GpuSyclIntel";
        };
    } // namespace detail

    //! The SYCL device manager.
    using PlatformGpuSyclIntel = PlatformGenericSycl<TagGpuSyclIntel>;
} // namespace alpaka

#endif
