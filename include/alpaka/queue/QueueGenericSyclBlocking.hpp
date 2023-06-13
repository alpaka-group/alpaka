/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include "alpaka/queue/sycl/QueueGenericSyclBase.hpp"

#    include <memory>
#    include <utility>

namespace alpaka
{
    template<typename TDev>
    using QueueGenericSyclBlocking = detail::QueueGenericSyclBase<TDev, true>;
} // namespace alpaka

#endif
