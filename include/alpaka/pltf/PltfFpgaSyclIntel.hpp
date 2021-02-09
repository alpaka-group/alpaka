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
#include <alpaka/pltf/PltfGenericSycl.hpp>
#include <alpaka/dev/DevGenericSycl.hpp>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace alpaka
{
    //#############################################################################
    //! The SYCL device manager.
    class PltfFpgaSyclIntel : public PltfGenericSycl
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST PltfFpgaSyclIntel() = delete;

#ifdef ALPAKA_FPGA_EMULATION
        using selector = cl::sycl::INTEL::fpga_emulator_selector;
#else
        using selector = cl::sycl::INTEL::fpga_selector;
#endif
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device manager device type trait specialization.
        template<>
        struct DevType<PltfFpgaSyclIntel>
        {
            using type = DevGenericSycl<PltfFpgaSyclIntel>; // = DevFpgaSyclIntel
        };
    }
}

#endif
