/* Copyright 2022 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/atomic/Op.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/mem/fence/Traits.hpp"
#include "alpaka/mem/order/MemOrderCuda.hpp"
#include "alpaka/mem/order/MemOrderHip.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    //! The GPU CUDA/HIP accelerator atomic ops.
    //
    //  Atomics can be used in the hierarchy level grids, blocks and threads.
    //  Atomics are not guaranteed to be safe between devices.
    class AtomicUniformCudaHipBuiltIn
    {
    };

    //! Note for CUDA
    // We have scoped atomics instructions available from SM70, which can be used alongside weaker fences
    // https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#atomics-application-binary-interface
    // We dont do this as it would require mapping types to the correct PTX atomic call which will be very verbose
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom

} // namespace alpaka

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !ALPAKA_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !ALPAKA_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

//! clang is providing a builtin for different atomic functions even if these is not supported for architectures < 6.0
#        define CLANG_CUDA_PTX_WORKAROUND                                                                             \
            (ALPAKA_COMP_CLANG && ALPAKA_LANG_CUDA && ALPAKA_ARCH_PTX < ALPAKA_VERSION_NUMBER(6, 0, 0))

#        if defined(ALPAKA_ARCH_PTX) || defined(ALPAKA_ARCH_AMD)
namespace detail
{
    // Generic atomic wrapper that handles memory ordering with fences
    template<typename TAtomicFunc, typename T, alpaka::MemoryOrder TMemOrder, alpaka::MemoryScope TMemScope>
    [[maybe_unused]] static constexpr __device__ T atomicOrderEmulated(
        [[maybe_unused]] auto const& acc,
        TAtomicFunc atomic_func,
        T* address,
        T value,
        TMemOrder,
        [[maybe_unused]] TMemScope scope)
    {
        if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::Relaxed>)
        {
            return atomic_func(address, value);
        }
        else if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::Acquire>)
        {
            auto ret = atomic_func(address, value);
            mem_fence(acc, alpaka::mem_order::acquire, scope);
            return ret;
        }
        else if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::Release>)
        {
            mem_fence(acc, alpaka::mem_order::release, scope);
            return atomic_func(address, value);
        }
        else if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::AcqRel>)
        {
            mem_fence(acc, alpaka::mem_order::release, scope);
            auto ret = atomic_func(address, value);
            mem_fence(acc, alpaka::mem_order::acquire, scope);
            return ret;
        }
        else
        { // sequentially consistent
            mem_fence(acc, alpaka::mem_order::seq_cst, scope);
            auto ret = atomic_func(address, value);
            // acquire fence is sufficient here for seq_cst guarantee
            // https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#atomics-application-binary-interface
            // TODO find documentation that this is sufficient for HIP
            mem_fence(acc, alpaka::mem_order::acquire, scope);
            return ret;
        }
    }

    // Generic atomic wrapper that handles memory ordering with fences
    template<typename TAtomicFunc, typename T, alpaka::MemoryOrder TMemOrder, alpaka::MemoryScope TMemScope>
    [[maybe_unused]] static constexpr __device__ T atomicCASOrderEmulated(
        [[maybe_unused]] auto const& acc,
        TAtomicFunc atomic_func,
        T* address,
        T compare,
        T value,
        [[maybe_unused]] TMemOrder order,
        [[maybe_unused]] TMemScope scope)
    {
        if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::Relaxed>)
        {
            return atomic_func(address, compare, value);
        }
        else if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::Acquire>)
        {
            auto ret = atomic_func(address, compare, value);
            mem_fence(acc, alpaka::mem_order::acquire, scope);
            return ret;
        }
        else if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::Release>)
        {
            mem_fence(acc, alpaka::mem_order::release, scope);
            return atomic_func(address, compare, value);
        }
        else if constexpr(std::is_same_v<TMemOrder, alpaka::mem_order::AcqRel>)
        {
            mem_fence(acc, alpaka::mem_order::release, scope);
            auto ret = atomic_func(address, compare, value);
            mem_fence(acc, alpaka::mem_order::acquire, scope);
            return ret;
        }
        else
        { // sequentially consistent
            mem_fence(acc, alpaka::mem_order::seq_cst, scope);
            auto ret = atomic_func(address, compare, value);
            // acquire fence is sufficient here for seq_cst guarantee
            // https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#atomics-application-binary-interface
            // TODO find documentation that this is sufficient for HIP
            mem_fence(acc, alpaka::mem_order::acquire, scope);
            return ret;
        }
    }


} // namespace detail

//! These types must be in the global namespace for checking existence of respective functions in global namespace via
//! SFINAE, so we use inline namespace.
inline namespace alpakaGlobal
{
    //! Provide an interface to builtin atomic functions.
    //
    // To check for the existence of builtin functions located in the global namespace :: directly.
    // This would not be possible without having these types in global namespace.
    // If the functor is inheriting from std::false_type an signature is explicitly not available. This can be used to
    // explicitly disable builtin function in case the builtin is broken.
    // If the functor is inheriting from std::true_type a specialization must implement one of the following
    // interfaces.
    // \code{.cpp}
    //    // interface for all atomics except atomicCas
    //    __device__ static T atomic( T* add, T value, TMemOrder order);
    //    // interface for atomicCas only
    //    __device__ static T atomic( T* add, T compare, T value, TMemOrder order);
    // \endcode
    template<typename TOp, typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy, typename TSfinae = void>
    struct AlpakaBuiltInAtomic : std::false_type
    {
    };

    // Cas.
    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicCas,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<
            decltype(atomicCAS(alpaka::core::declval<T*>(), alpaka::core::declval<T>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T compare, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_compare_exchange(
                add,
                &compare,
                value,
                alpaka::MemOrderCuda::get(order),
                alpaka::MemOrderCuda::get(order),
                __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicCASOrderEmulated(
                acc,
                [](T* addr, T cmp, T val) { return atomicCAS(addr, cmp, val); },
                add,
                compare,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_compare_exchange_strong(
                add,
                &compare,
                value,
                alpaka::MemOrderHip::get(order),
                alpaka::MemOrderHip::get(order),
                __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicCas,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicCAS_block(
            alpaka::core::declval<T*>(),
            alpaka::core::declval<T>(),
            alpaka::core::declval<T>()))>> : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T compare, T value, TMemOrder order)
        {
#                if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_compare_exchange(
                add,
                &compare,
                value,
                alpaka::MemOrderCuda::get(order),
                alpaka::MemOrderCuda::get(order),
                __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicCASOrderEmulated(
                acc,
                [](T* addr, T cmp, T val) { return atomicCAS_block(addr, cmp, val); },
                add,
                compare,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_compare_exchange_strong(
                add,
                &compare,
                value,
                alpaka::MemOrderHip::get(order),
                alpaka::MemOrderHip::get(order),
                __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif


    // Add.
    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicAdd,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicAdd(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_add(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicAdd(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_add(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };


#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicAdd,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicAdd_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_add(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicAdd_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_add(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif

#            if CLANG_CUDA_PTX_WORKAROUND
    // clang is providing a builtin for atomicAdd even if these is not supported by the current architecture
    template<alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::AtomicAdd, double, TMemOrder, THierarchy> : std::false_type
    {
    };
#            endif

#            if(ALPAKA_LANG_HIP)
    // HIP shows bad performance with builtin atomicAdd(float*,float) for the hierarchy threads therefore we do not
    // call the buildin method and instead use the atomicCAS emulation. For details see:
    // https://github.com/alpaka-group/alpaka/issues/1657
    template<alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<alpaka::AtomicAdd, float, TMemOrder, alpaka::hierarchy::Threads> : std::false_type
    {
    };
#            endif

    // Sub.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicSub,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicSub(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_sub(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicSub(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
#                    if __has_builtin(__hip_atomic_fetch_sub)
            return __hip_atomic_fetch_sub(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                    else
            // emulate with fetch add
            if constexpr(std::is_unsigned_v<T>)
            {
                using signed_t = std::make_signed_t<T>;
                return __hip_atomic_fetch_add(
                    add,
                    static_cast<T>(-static_cast<signed_t>(value)),
                    alpaka::MemOrderHip::get(order),
                    __HIP_MEMORY_SCOPE_AGENT);
            }
            else
            {
                return __hip_atomic_fetch_add(add, -value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
            }
#                    endif
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicSub,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicSub_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_sub(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicSub_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
#                        if __has_builtin(__hip_atomic_fetch_sub)
            return __hip_atomic_fetch_sub(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                        else
            // emulate with fetch add
            if constexpr(std::is_unsigned_v<T>)
            {
                using signed_t = std::make_signed_t<T>;
                return __hip_atomic_fetch_add(
                    add,
                    static_cast<T>(-static_cast<signed_t>(value)),
                    alpaka::MemOrderHip::get(order),
                    __HIP_MEMORY_SCOPE_WORKGROUP);
            }
            else
            {
                return __hip_atomic_fetch_add(
                    add,
                    -value,
                    alpaka::MemOrderHip::get(order),
                    __HIP_MEMORY_SCOPE_WORKGROUP);
            }
#                        endif
#                    endif
#                endif
        }
    };
#            endif

    // Min.
    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicMin,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicMin(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED

#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_min(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicMin(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_min(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicMin,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicMin_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_min(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicMin_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_min(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif

// disable HIP atomicMin: see https://github.com/ROCm-Developer-Tools/hipamd/pull/40
#            if(ALPAKA_LANG_HIP)
    template<alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMin, float, TMemOrder, THierarchy> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMin, float, TMemOrder, alpaka::hierarchy::Threads> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMin, double, TMemOrder, THierarchy> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMin, double, TMemOrder, alpaka::hierarchy::Threads> : std::false_type
    {
    };

#                if !__has_builtin(__hip_atomic_compare_exchange_strong)
    template<alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMin, unsigned long long, TMemOrder, THierarchy> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMin, unsigned long long, TMemOrder, alpaka::hierarchy::Threads>
        : std::false_type
    {
    };
#                endif
#            endif

    // Max.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicMax,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicMax(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_max(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicMax(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_max(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicMax,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicMax_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_max(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicMax_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_max(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif

    // disable HIP atomicMax: see https://github.com/ROCm-Developer-Tools/hipamd/pull/40
#            if(ALPAKA_LANG_HIP)
    template<alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMax, float, TMemOrder, THierarchy> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMax, float, TMemOrder, alpaka::hierarchy::Threads> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMax, double, TMemOrder, THierarchy> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMax, double, TMemOrder, alpaka::hierarchy::Threads> : std::false_type
    {
    };

#                if !__has_builtin(__hip_atomic_compare_exchange_strong)
    template<alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMax, unsigned long long, TMemOrder, THierarchy> : std::false_type
    {
    };

    template<alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<alpaka::AtomicMax, unsigned long long, TMemOrder, alpaka::hierarchy::Threads>
        : std::false_type
    {
    };
#                endif
#            endif


    // Exch.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicExch,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicExch(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_exchange(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicExch(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_exchange(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicExch,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicExch_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_exchange(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicExch_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_exchange(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif

    // Inc.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicInc,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicInc(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
            // Note atomicInc doesn't have a direct __nv/hip_atomic equivalent, so we use the fence emulation
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicInc(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicInc,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicInc_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
            // Note atomicInc_block doesn't have a direct __nv/hip_atomic equivalent, so we use the fence emulation
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicInc_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
        }
    };
#            endif

    // Dec.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicDec,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicDec(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
            // Note atomicDec doesn't have a direct __nv/hip_atomic or equivalent, so we use the fence emulation
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicDec(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicDec,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicDec_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
            // Note atomicDec_block doesn't have a direct __nv/hip_atomic equivalent, so we use the fence emulation
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicDec_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
        }
    };
#            endif

    // And.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicAnd,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicAnd(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_and(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicAnd(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_and(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicAnd,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicAnd_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_and(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicAnd_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_and(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif

    // Or.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicOr,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicOr(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_or(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicOr(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_or(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicOr,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicOr_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_or(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicOr_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_or(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif

    // Xor.

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicXor,
        T,
        TMemOrder,
        THierarchy,
        typename std::void_t<decltype(atomicXor(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#            if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#                if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_xor(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_DEVICE);
#                else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicXor(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::device);
#                endif
#            else
#                if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_xor(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_AGENT);
#                endif
#            endif
        }
    };

#            if !CLANG_CUDA_PTX_WORKAROUND
    template<typename T, alpaka::MemoryOrder TMemOrder>
    struct AlpakaBuiltInAtomic<
        alpaka::AtomicXor,
        T,
        TMemOrder,
        alpaka::hierarchy::Threads,
        typename std::void_t<decltype(atomicXor_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
        : std::true_type
    {
        static __device__ T atomic([[maybe_unused]] auto const& acc, T* add, T value, TMemOrder order)
        {
#                ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#                    if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX
            return __nv_atomic_fetch_xor(add, value, alpaka::MemOrderCuda::get(order), __NV_THREAD_SCOPE_BLOCK);
#                    else
            return detail::atomicOrderEmulated(
                acc,
                [](T* addr, T val) { return atomicXor_block(addr, val); },
                add,
                value,
                order,
                alpaka::memory_scope::block);
#                    endif
#                else
#                    if ALPAKA_LANG_HIP
            return __hip_atomic_fetch_xor(add, value, alpaka::MemOrderHip::get(order), __HIP_MEMORY_SCOPE_WORKGROUP);
#                    endif
#                endif
        }
    };
#            endif

} // namespace alpakaGlobal
#        endif

#        undef CLANG_CUDA_PTX_WORKAROUND
#    endif

#endif
