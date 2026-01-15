/* Copyright 2022 Felice Pantaleo, Andrea Bocci, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)                    \
    || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)                   \
    || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)

#    include "alpaka/atomic/Traits.hpp"
#    include "alpaka/core/Config.hpp"
#    include "alpaka/mem/order/MemOrderStl.hpp"
#    include "alpaka/mem/order/MemoryOrder.hpp"

#    include <array>
#    include <atomic>
#    include <type_traits>

#    ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#        ifndef ALPAKA_HAS_STD_ATOMIC_REF
#            include "alpaka/mem/order/MemOrderBoost.hpp"

#            include <boost/atomic.hpp>
#        endif

namespace alpaka
{
    namespace detail
    {
#        if defined(ALPAKA_HAS_STD_ATOMIC_REF)
        template<typename T>
        using atomic_ref = std::atomic_ref<T>;
        using MemOrderCpu = MemOrderStl;
#        else
        template<typename T>
        using atomic_ref = boost::atomic_ref<T>;
        using MemOrderCpu = MemOrderBoost;
#        endif
    } // namespace detail

    //! The atomic ops based on atomic_ref for CPU accelerators.
    //
    //  Atomics can be used in the grids, blocks and threads hierarchy levels.
    //

    class AtomicAtomicRef
    {
    };

    template<typename T>
    void isSupportedByAtomicAtomicRef()
    {
        static_assert(
            std::is_trivially_copyable_v<T> && alpaka::detail::atomic_ref<T>::required_alignment <= alignof(T),
            "Type not supported by AtomicAtomicRef, please recompile defining "
            "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
    }

    namespace trait
    {
        //! The CPU accelerators AtomicAdd.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                return ref.fetch_add(value, detail::MemOrderCpu::get(order));
            }
        };

        //! The CPU accelerators AtomicSub.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                return ref.fetch_sub(value, detail::MemOrderCpu::get(order));
            }
        };

        //! The CPU accelerators AtomicMin.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                T old = ref.load(detail::MemOrderCpu::get(order));
                T result = old;
                result = std::min(result, value);
                while(!ref.compare_exchange_weak(old, result, detail::MemOrderCpu::get(order)))
                {
                    result = old;
                    result = std::min(result, value);
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicMax.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                T old = ref.load(detail::MemOrderCpu::get(order));
                T result = old;
                result = std::max(result, value);
                while(!ref.compare_exchange_weak(old, result, detail::MemOrderCpu::get(order)))
                {
                    result = old;
                    result = std::max(result, value);
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicExch.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                T old = ref.load(detail::MemOrderCpu::get(order));
                T result = value;
                while(!ref.compare_exchange_weak(old, result, detail::MemOrderCpu::get(order)))
                {
                    result = value;
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicInc.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                T old = ref.load(detail::MemOrderCpu::get(order));
                T result = ((old >= value) ? 0 : static_cast<T>(old + 1));
                while(!ref.compare_exchange_weak(old, result, detail::MemOrderCpu::get(order)))
                {
                    result = ((old >= value) ? 0 : static_cast<T>(old + 1));
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicDec.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                T old = ref.load(detail::MemOrderCpu::get(order));
                T result = (old == static_cast<T>(0) || old > value) ? value : (old - static_cast<T>(1));
                while(!ref.compare_exchange_weak(old, result, detail::MemOrderCpu::get(order)))
                {
                    result = (old == static_cast<T>(0) || old > value) ? value : (old - static_cast<T>(1));
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicAnd.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                return ref.fetch_and(value, detail::MemOrderCpu::get(order));
            }
        };

        //! The CPU accelerators AtomicOr.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                return ref.fetch_or(value, detail::MemOrderCpu::get(order));
            }
        };

        //! The CPU accelerators AtomicXor.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value, TMemOrder order)
                -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                return ref.fetch_xor(value, detail::MemOrderCpu::get(order));
            }
        };

        //! The CPU accelerators AtomicCas.
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicAtomicRef, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(
                AtomicAtomicRef const&,
                T* const addr,
                T const& compare,
                T const& value,
                TMemOrder order) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                alpaka::detail::atomic_ref<T> ref(*addr);
                T old = ref.load(detail::MemOrderCpu::get(order));
                T result;
                do
                {
#        if ALPAKA_COMP_GNUC || ALPAKA_COMP_CLANG
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wfloat-equal"
#        endif
                    result = ((old == compare) ? value : old);
#        if ALPAKA_COMP_GNUC || ALPAKA_COMP_CLANG
#            pragma GCC diagnostic pop
#        endif
                } while(!ref.compare_exchange_weak(old, result, detail::MemOrderCpu::get(order)));
                return old;
            }
        };
    } // namespace trait
} // namespace alpaka

#    endif
#endif
