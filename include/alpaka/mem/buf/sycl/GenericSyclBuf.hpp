/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/sycl/ConstBufGenericSycl.hpp"
#include "alpaka/mem/buf/sycl/MutBufGenericSycl.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    using BufGenericSycl = MutBufGenericSycl<
        ConstBufGenericSycl,
        detail::BufSyclImpl<TElem, TDim, TIdx, TTag>,
        DevGenericSycl<TTag>,
        TElem,
        TDim,
        TIdx,
        TTag>;
} // namespace alpaka

#    include "alpaka/mem/buf/sycl/Copy.hpp"
#    include "alpaka/mem/buf/sycl/Set.hpp"
#    include "alpaka/mem/buf/sycl/traits/ConstBufGenericSyclTraits.hpp"
#    include "alpaka/mem/buf/sycl/traits/MutBufGenericSyclTraits.hpp"

#endif
