/* Copyright 2022 Benjamin Worpitz, René Widera, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/atomic/AtomicUniformCudaHip.hpp"
#include "alpaka/atomic/Op.hpp"
#include "alpaka/atomic/Traits.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/core/Decay.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

#include <limits>
#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !ALPAKA_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !ALPAKA_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

namespace alpaka::trait
{
    namespace detail
    {
        struct EmulationBase
        {
            //! reinterprets an address as an 32bit value for atomicCas emulation usage
            template<typename TAddressType>
            static __device__ auto reinterpretAddress(TAddressType* address)
                -> std::enable_if_t<sizeof(TAddressType) == 4u, unsigned int*>
            {
                return reinterpret_cast<unsigned int*>(address);
            }

            //! reinterprets a address as an 64bit value for atomicCas emulation usage
            template<typename TAddressType>
            static __device__ auto reinterpretAddress(TAddressType* address)
                -> std::enable_if_t<sizeof(TAddressType) == 8u, unsigned long long int*>
            {
                return reinterpret_cast<unsigned long long int*>(address);
            }

            //! reinterprets a value to be usable for the atomicCAS emulation
            template<typename T_Type>
            static __device__ auto reinterpretValue(T_Type value)
            {
                return *reinterpretAddress(&value);
            }
        };

        //! Emulate atomic
        //
        // The default implementation will emulate all atomic functions with atomicCAS.
        template<
            typename TOp,
            typename TAtomic,
            typename T,
            alpaka::MemoryOrder TMemOrder,
            typename THierarchy,
            typename TSfinae = void,
            typename TDefer = void>
        struct EmulateAtomic : private EmulationBase
        {
        public:
            static __device__ auto atomic(
                alpaka::AtomicUniformCudaHipBuiltIn const& ctx,
                T* const addr,
                T const& value,
                TMemOrder order) -> T
            {
                auto* const addressAsIntegralType = reinterpretAddress(addr);
                using EmulatedType = std::decay_t<decltype(*addressAsIntegralType)>;

                // Emulating atomics with atomicCAS is mentioned in the programming guide too.
                // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
#        if ALPAKA_LANG_HIP
#            if __has_builtin(__hip_atomic_load)
                EmulatedType old{__hip_atomic_load(addressAsIntegralType, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)};
#            else
                EmulatedType old{__atomic_load_n(addressAsIntegralType, __ATOMIC_RELAXED)};
#            endif
#        else
                EmulatedType old{*addressAsIntegralType};
#        endif
                EmulatedType assumed;
                do
                {
                    assumed = old;
                    T v = *(reinterpret_cast<T*>(&assumed));
                    TOp{}(&v, value);
                    using Cas = alpaka::trait::AtomicOp<
                        alpaka::AtomicCas,
                        alpaka::AtomicUniformCudaHipBuiltIn,
                        EmulatedType,
                        TMemOrder,
                        THierarchy>;
                    old = Cas::atomicOp(ctx, addressAsIntegralType, assumed, reinterpretValue(v), order);
                    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                } while(assumed != old);
                return *(reinterpret_cast<T*>(&old));
            }
        };

        //! Emulate AtomicCas with equivalent unisigned integral type
        template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
        struct EmulateAtomic<alpaka::AtomicCas, alpaka::AtomicUniformCudaHipBuiltIn, T, TMemOrder, THierarchy>
            : private EmulationBase
        {
            static __device__ auto atomic(
                alpaka::AtomicUniformCudaHipBuiltIn const& ctx,
                T* const addr,
                T const& compare,
                T const& value,
                TMemOrder order) -> T
            {
                auto* const addressAsIntegralType = reinterpretAddress(addr);
                using EmulatedType = std::decay_t<decltype(*addressAsIntegralType)>;
                EmulatedType reinterpretedCompare = reinterpretValue(compare);
                EmulatedType reinterpretedValue = reinterpretValue(value);

                auto old = alpaka::trait::AtomicOp<
                    alpaka::AtomicCas,
                    alpaka::AtomicUniformCudaHipBuiltIn,
                    EmulatedType,
                    TMemOrder,
                    THierarchy>::atomicOp(ctx, addressAsIntegralType, reinterpretedCompare, reinterpretedValue, order);

                return *(reinterpret_cast<T*>(&old));
            }
        };

        //! Emulate AtomicSub with atomicAdd
        template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
        struct EmulateAtomic<alpaka::AtomicSub, alpaka::AtomicUniformCudaHipBuiltIn, T, TMemOrder, THierarchy>
        {
            static __device__ auto atomic(
                alpaka::AtomicUniformCudaHipBuiltIn const& ctx,
                T* const addr,
                T const& value,
                TMemOrder order) -> T
            {
                return alpaka::trait::
                    AtomicOp<alpaka::AtomicAdd, alpaka::AtomicUniformCudaHipBuiltIn, T, TMemOrder, THierarchy>::
                        atomicOp(ctx, addr, -value, order);
            }
        };

        //! AtomicDec can not be implemented for floating point types!
        template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
        struct EmulateAtomic<
            alpaka::AtomicDec,
            alpaka::AtomicUniformCudaHipBuiltIn,
            T,
            TMemOrder,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(alpaka::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::AtomicDec> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicInc can not be implemented for floating point types!
        template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
        struct EmulateAtomic<
            alpaka::AtomicInc,
            alpaka::AtomicUniformCudaHipBuiltIn,
            T,
            TMemOrder,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(alpaka::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::AtomicInc> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicAnd can not be implemented for floating point types!
        template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
        struct EmulateAtomic<
            alpaka::AtomicAnd,
            alpaka::AtomicUniformCudaHipBuiltIn,
            T,
            TMemOrder,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(alpaka::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::AtomicAnd> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicOr can not be implemented for floating point types!
        template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
        struct EmulateAtomic<
            alpaka::AtomicOr,
            alpaka::AtomicUniformCudaHipBuiltIn,
            T,
            TMemOrder,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(alpaka::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::AtomicOr> is not supported for floating point data types!");
                return T{};
            }
        };

        //! AtomicXor can not be implemented for floating point types!
        template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
        struct EmulateAtomic<
            alpaka::AtomicXor,
            alpaka::AtomicUniformCudaHipBuiltIn,
            T,
            TMemOrder,
            THierarchy,
            std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static __device__ auto atomic(alpaka::AtomicUniformCudaHipBuiltIn const&, T* const, T const&) -> T
            {
                static_assert(
                    !sizeof(T),
                    "EmulateAtomic<alpaka::AtomicXor> is not supported for floating point data types!");
                return T{};
            }
        };

    } // namespace detail

    //! Generic atomic implementation
    //
    // - unsigned long int will be redirected to unsigned long long int or unsigned int implementation depending if
    //   unsigned long int is a 64 or 32bit data type.
    // - Atomics which are not available as builtin atomic will be emulated.
    template<typename TOp, typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AtomicOp<TOp, AtomicUniformCudaHipBuiltIn, T, TMemOrder, THierarchy>
    {
        static __device__ auto atomicOp(
            auto const& ctx,
            [[maybe_unused]] T* const addr,
            [[maybe_unused]] T const& value,
            TMemOrder order) -> T
        {
            static_assert(
                sizeof(T) == 4u || sizeof(T) == 8u,
                "atomicOp<TOp, AtomicUniformCudaHipBuiltIn, T>(atomic, addr, value) is not supported! Only 64 and "
                "32bit atomics are supported.");

            if constexpr(::AlpakaBuiltInAtomic<TOp, T, TMemOrder, THierarchy>::value)
                return ::AlpakaBuiltInAtomic<TOp, T, TMemOrder, THierarchy>::atomic(ctx, addr, value, order);

            else if constexpr(std::is_same_v<unsigned long int, T>)
            {
                if constexpr(sizeof(T) == 4u && ::AlpakaBuiltInAtomic<TOp, unsigned int, TMemOrder, THierarchy>::value)
                    return ::AlpakaBuiltInAtomic<TOp, unsigned int, TMemOrder, THierarchy>::atomic(
                        ctx,
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(value),
                        order);
                else if constexpr(
                    sizeof(T) == 8u
                    && ::AlpakaBuiltInAtomic<TOp, unsigned long long int, TMemOrder, THierarchy>::value) // LP64
                {
                    return ::AlpakaBuiltInAtomic<TOp, unsigned long long int, TMemOrder, THierarchy>::atomic(
                        ctx,
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(value),
                        order);
                }
            }

            return detail::EmulateAtomic<TOp, AtomicUniformCudaHipBuiltIn, T, TMemOrder, THierarchy>::atomic(
                ctx,
                addr,
                value,
                order);
        }
    };

    template<typename T, alpaka::MemoryOrder TMemOrder, typename THierarchy>
    struct AtomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, T, TMemOrder, THierarchy>
    {
        static __device__ auto atomicOp(
            [[maybe_unused]] auto const& ctx,
            [[maybe_unused]] T* const addr,
            [[maybe_unused]] T const& compare,
            [[maybe_unused]] T const& value,
            TMemOrder order) -> T
        {
            static_assert(
                sizeof(T) == 4u || sizeof(T) == 8u,
                "atomicOp<AtomicCas, AtomicUniformCudaHipBuiltIn, T>(atomic, addr, compare, value) is not "
                "supported! Only 64 and "
                "32bit atomics are supported.");

            if constexpr(::AlpakaBuiltInAtomic<AtomicCas, T, TMemOrder, THierarchy>::value)
                return ::AlpakaBuiltInAtomic<AtomicCas, T, TMemOrder, THierarchy>::atomic(
                    ctx,
                    addr,
                    compare,
                    value,
                    order);

            else if constexpr(std::is_same_v<unsigned long int, T>)
            {
                if constexpr(
                    sizeof(T) == 4u && ::AlpakaBuiltInAtomic<AtomicCas, unsigned int, TMemOrder, THierarchy>::value)
                    return ::AlpakaBuiltInAtomic<AtomicCas, unsigned int, TMemOrder, THierarchy>::atomic(
                        ctx,
                        reinterpret_cast<unsigned int*>(addr),
                        static_cast<unsigned int>(compare),
                        static_cast<unsigned int>(value),
                        order);
                else if constexpr(
                    sizeof(T) == 8u
                    && ::AlpakaBuiltInAtomic<AtomicCas, unsigned long long int, TMemOrder, THierarchy>::value) // LP64
                {
                    return ::AlpakaBuiltInAtomic<AtomicCas, unsigned long long int, TMemOrder, THierarchy>::atomic(
                        ctx,
                        reinterpret_cast<unsigned long long int*>(addr),
                        static_cast<unsigned long long int>(compare),
                        static_cast<unsigned long long int>(value),
                        order);
                }
            }

            return detail::EmulateAtomic<AtomicCas, AtomicUniformCudaHipBuiltIn, T, TMemOrder, THierarchy>::atomic(
                ctx,
                addr,
                compare,
                value,
                order);
        }
    };
} // namespace alpaka::trait
#    endif
#endif
