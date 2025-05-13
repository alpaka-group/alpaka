/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/mem/buf/cpu/ConstBufCpu.hpp"
#include "alpaka/mem/buf/cpu/MutBufCpu.hpp"

namespace alpaka
{
    template<typename TElem, typename TDim, typename TIdx>
    using BufCpu = MutBufCpu<ConstBufCpu, alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>, DevCpu, TElem, TDim, TIdx>;
} // namespace alpaka

#include "alpaka/mem/buf/cpu/Copy.hpp"
#include "alpaka/mem/buf/cpu/Set.hpp"
#include "alpaka/mem/buf/cpu/traits/ConstBufCpuTraits.hpp"
#include "alpaka/mem/buf/cpu/traits/MutBufCpuTraits.hpp"
