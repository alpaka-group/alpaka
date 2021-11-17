/* Copyright 2021 Felice Pantaleo, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <boost/atomic.hpp>

#include <array>
#include <type_traits>

namespace alpaka
{
    //! The CPU threads accelerator atomic ops based on atomic_ref.
    //
    //  Atomics can be used in the grids, blocks and threads hierarchy levels.
    //

    class AtomicAtomicRef
    {
    };

    namespace traits
    {
        //! The CPU threads accelerator AtomicAdd.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                return ref.fetch_add(value);
            }
        };

        //! The CPU threads accelerator AtomicSub.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                return ref.fetch_sub(value);
            }
        };

        //! The CPU threads accelerator AtomicMin.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMin, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = old;
                result = std::min(result, value);
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = old;
                    result = std::min(result, value);
                }
                return old;
            }
        };

        //! The CPU threads accelerator AtomicMax.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicMax, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = old;
                result = std::max(result, value);
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = old;
                    result = std::max(result, value);
                }
                return old;
            }
        };

        //! The CPU threads accelerator AtomicExch.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = value;
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = value;
                }
                return old;
            }
        };

        //! The CPU threads accelerator AtomicInc.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicInc, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = ((old >= value) ? 0 : static_cast<T>(old + 1));
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = ((old >= value) ? 0 : static_cast<T>(old + 1));
                }
                return old;
            }
        };

        //! The CPU threads accelerator AtomicDec.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicDec, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = ((old >= value) ? 0 : static_cast<T>(old - 1));
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = ((old >= value) ? 0 : static_cast<T>(old - 1));
                }
                return old;
            }
        };

        //! The CPU threads accelerator AtomicAnd.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                return ref.fetch_and(value);
            }
        };

        //! The CPU threads accelerator AtomicOr.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                return ref.fetch_or(value);
            }
        };

        //! The CPU threads accelerator AtomicXor.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const& atomic, T* const addr, T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                return ref.fetch_xor(value);
            }
        };

        //! The CPU threads accelerator AtomicCas.
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicCas, AtomicAtomicRef, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(
                AtomicAtomicRef const& atomic,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                static_assert(
                    std::is_trivially_copyable_v<T> && boost::atomic_ref<T>::required_alignment <= alignof(T),
                    "Type not supported by AtomicAtomicRef, please recompile defining "
                    "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
                boost::atomic_ref<T> ref(*addr);
                T old = ref;
                T result;
                do
                {
                    result = ((old == compare) ? value : old);
                } while(!ref.compare_exchange_weak(old, result));
                return old;
            }
        };
    } // namespace traits
} // namespace alpaka
