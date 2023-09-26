/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Luca Ferragina <luca.ferragina@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 *
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
        struct IntelFpgaSelector final
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
        };
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
    } // namespace detail

    //! The SYCL device manager.
    using PlatformFpgaSyclIntel = PlatformGenericSycl<detail::IntelFpgaSelector>;
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL device manager device type trait specialization.
    template<>
    struct DevType<PlatformFpgaSyclIntel>
    {
        using type = DevGenericSycl<PlatformFpgaSyclIntel>; // = DevFpgaSyclIntel
    };
} // namespace alpaka::trait

#endif
