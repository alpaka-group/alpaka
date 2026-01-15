/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/order/MemoryOrder.hpp"

#include <concepts>

#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#    ifndef ALPAKA_HAS_STD_ATOMIC_REF
#        include <boost/memory_order.hpp>

namespace alpaka
{
    struct MemOrderBoost
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return boost::memory_order::seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return boost::memory_order::acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return boost::memory_order::release;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return boost::memory_order_acquire;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return boost::memory_order_relaxed;
            }
        }
    };

} // namespace alpaka
#    endif
#endif
