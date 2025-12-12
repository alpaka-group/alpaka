/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Config.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

#include <concepts>

#if defined ALPAKA_ACC_GPU_CUDA_ENABLED && ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX

namespace alpaka
{
    struct MemOrderCuda
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return __NV_ATOMIC_SEQ_CST;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return __NV_ATOMIC_ACQ_REL;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return __NV_ATOMIC_RELEASE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return __NV_ATOMIC_ACQUIRE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return __NV_ATOMIC_RELAXED;
            }
        }
    };

} // namespace alpaka

#endif
