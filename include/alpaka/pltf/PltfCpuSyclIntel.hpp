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

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevGenericSycl.hpp>
#include <alpaka/pltf/PltfGenericSycl.hpp>

#include <CL/sycl.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace alpaka
{
    namespace detail
    {
        // Prevent clang from annoying us with warnings about emitting too many vtables. These are discarded by
        // the linker anyway.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"
        struct intel_cpu_selector : cl::sycl::device_selector
        {
            auto operator()(const cl::sycl::device& dev) const -> int override
            {
                const auto vendor = dev.get_info<cl::sycl::info::device::vendor>();
                const auto is_intel_cpu = (vendor.find("Intel(R) Corporation") != std::string::npos) && dev.is_cpu();

                return is_intel_cpu ? 1 : -1;
            }
        };
#pragma clang diagnostic pop
    }

    //#############################################################################
    //! The SYCL device manager.
    class PltfCpuSyclIntel : public PltfGenericSycl
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST PltfCpuSyclIntel() = delete;

        using selector = detail::intel_cpu_selector;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device manager device type trait specialization.
        template<>
        struct DevType<PltfCpuSyclIntel>
        {
            using type = DevGenericSycl<PltfCpuSyclIntel>; // = DevCpuSyclIntel
        };
    }
}

#endif
