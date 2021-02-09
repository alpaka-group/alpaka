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

#include <alpaka/dev/DevCpuSyclIntel.hpp>
#include <alpaka/mem/buf/BufGenericSycl.hpp>

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufCpuSyclIntel = BufGenericSycl<TElem, TDim, TIdx, DevCpuSyclIntel>;
}

#endif
