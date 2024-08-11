/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/platform/PlatformGenericSycl.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)

#    include <sycl/sycl.hpp>

#    include <string>

namespace alpaka
{
    namespace detail
    {
        // Prevent clang from annoying us with warnings about emitting too many vtables. These are discarded by the
        // linker anyway.
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wweak-vtables"
#    endif
        template<>
        struct SYCLDeviceSelector<TagFpgaSyclIntel>
        {
#    ifdef ALPAKA_FPGA_EMULATION
            static constexpr auto platform_name = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
#    else
            static constexpr auto platform_name = "Intel(R) FPGA SDK for OpenCL(TM)";
#    endif

            auto operator()(sycl::device const& dev) const -> int
            {
                auto const& platform = dev.get_platform().get_info<sycl::info::platform::name>();
                auto const is_intel_fpga = dev.is_accelerator() && (platform == platform_name);

                return is_intel_fpga ? 1 : -1;
            }

            static constexpr char name[] = "FpgaSyclIntel";
        };
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
    } // namespace detail

    //! The SYCL device manager.
    using PlatformFpgaSyclIntel = PlatformGenericSycl<TagFpgaSyclIntel>;
} // namespace alpaka

#endif
