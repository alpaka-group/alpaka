/* Copyright 2022 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/atomic/AtomicOmpMacros.hpp"
#include "alpaka/atomic/Op.hpp"
#include "alpaka/atomic/Traits.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

#ifdef _OPENMP


namespace alpaka
{
    //! The OpenMP accelerators atomic ops.
    //
    //  Atomics can be used in the blocks and threads hierarchy levels.
    //  Atomics are not guaranteed to be safe between devices or grids.
    class AtomicOmpBuiltIn
    {
    };

    namespace trait
    {
// check for OpenMP 3.1+
// "omp atomic capture" is not supported before OpenMP 3.1
#    if _OPENMP >= 201107

        //! The OpenMP accelerators atomic operation: ADD
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
                ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, {
                    old = ref;
                    ref += value;
                });
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: SUB
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
                ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, {
                    old = ref;
                    ref -= value;
                });
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: EXCH
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
                // atomically update ref, but capture the original value in old
                ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, {
                    old = ref;
                    ref = value;
                });
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: AND
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
                ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, {
                    old = ref;
                    ref &= value;
                });
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: OR
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
                ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, {
                    old = ref;
                    ref |= value;
                });
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: XOR
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic push
#            pragma GCC diagnostic ignored "-Wconversion"
#        endif
                ALPAKA_OMP_ATOMIC_CAPTURE_ORDER(TMemOrder, {
                    old = ref;
                    ref ^= value;
                });
#        if ALPAKA_COMP_GNUC
#            pragma GCC diagnostic pop
#        endif
                return old;
            }
        };

#    endif // _OPENMP >= 201107

// check for OpenMP 5.1+
// "omp atomic compare" was introduced with OpenMP 5.1
#    if _OPENMP >= 202011

        //! The OpenMP accelerators atomic operation: Min
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
                // atomically update ref, but capture the original value in old
                ALPAKA_OMP_ATOMIC_CAPTURE_COMPARE_ORDER(TMemOrder, {
                    old = ref;
                    // Do not remove the curly brackets of the if body else
                    // icpx 2024.0 is not able to compile the atomics.
                    if(value < ref)
                    {
                        ref = value;
                    }
                });
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: Max
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T value, TMemOrder) -> T
            {
                T old;
                auto& ref(*addr);
                // atomically update ref, but capture the original value in old
                ALPAKA_OMP_ATOMIC_CAPTURE_COMPARE_ORDER(TMemOrder, {
                    old = ref;
                    // Do not remove the curly brackets of the if body else
                    // icpx 2024.0 is not able to compile the atomics.
                    if(value > ref)
                    {
                        ref = value;
                    }
                });
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: Inc
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                // TODO(bgruber): atomic increment with wrap around is not implementable in OpenMP 5.1
                T old;
#        pragma omp critical(AlpakaOmpAtomicOp)
                {
                    old = AtomicInc{}(addr, value);
                }
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: Dec
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                // TODO(bgruber): atomic decrement with wrap around is not implementable in OpenMP 5.1
                T old;
#        pragma omp critical(AlpakaOmpAtomicOp)
                {
                    old = AtomicDec{}(addr, value);
                }
                return old;
            }
        };

        //! The OpenMP accelerators atomic operation: Cas
        template<typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T compare, T value, TMemOrder)
                -> T
            {
                T old;
                auto& ref(*addr);
                // atomically update ref, but capture the original value in old
                ALPAKA_OMP_ATOMIC_CAPTURE_COMPARE_ORDER(TMemOrder, {
                    old = ref;
                    // Do not remove the curly brackets of the if body else
                    // icpx 2024.0 is not able to compile the atomics.
                    if(ref == compare)
                    {
                        ref = value;
                    }
                });
                return old;
            }
        };

#    else
        //! The OpenMP accelerators atomic operation
        //
        // generic implementations for operations where native atomics are not available
        template<typename TOp, typename T, MemoryOrder TMemOrder, typename THierarchy>
        struct AtomicOp<TOp, AtomicOmpBuiltIn, T, TMemOrder, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value, TMemOrder) -> T
            {
                T old;
                // \TODO: Currently not only the access to the same memory location is protected by a mutex but all
                // atomic ops on all threads.
#        pragma omp critical(AlpakaOmpAtomicOp)
                {
                    old = TOp()(addr, value);
                }
                return old;
            }

            ALPAKA_FN_HOST static auto atomicOp(
                AtomicOmpBuiltIn const&,
                T* const addr,
                T const& compare,
                T const& value,
                TMemOrder) -> T
            {
                T old;
                // \TODO: Currently not only the access to the same memory location is protected by a mutex but all
                // atomic ops on all threads.
#        pragma omp critical(AlpakaOmpAtomicOp2)
                {
                    old = TOp()(addr, compare, value);
                }
                return old;
            }
        };

#    endif // _OPENMP >= 202011

    } // namespace trait
} // namespace alpaka

#endif
