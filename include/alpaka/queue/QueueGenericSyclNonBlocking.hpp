/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/queue/sycl/QueueGenericSyclBase.hpp"

#include <memory>
#include <utility>

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka
{
    template<typename TDev>
    using QueueGenericSyclNonBlocking = detail::QueueGenericSyclBase<TDev, false>;
}

#endif
