/* Copyright 2022 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/internal/interface.hpp"
#include "alpaka/onAcc/scope.hpp"
#include "alpaka/operation.hpp"

#include <type_traits>

namespace alpaka::onAcc
{
    //! Executes the given operation atomically.
    //!
    //! \tparam T_Op The operation type.
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T_Op, typename T, typename T_Scope = scope::Device>
    constexpr auto atomicOp(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        static_assert(!std::is_same_v<T_Scope, scope::System>, "System scope is currently not supported.");
        auto atomicImpl = trait::getAtomicImpl(acc[object::exec], scope);
        return internalCompute::Atomic::Op<T_Op, ALPAKA_TYPEOF(atomicImpl), T, T_Scope>::atomicOp(
            atomicImpl,
            addr,
            value);
    }

    //! Executes the given operation atomically.
    //!
    //! \tparam T_Op The operation type.
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    template<typename T_Op, typename T, typename T_Scope = scope::Device>
    constexpr auto atomicOp(
        auto const& acc,
        T* const addr,
        T const& compare,
        T const& value,
        T_Scope const scope = T_Scope()) -> T
    {
        static_assert(!std::is_same_v<T_Scope, scope::System>, "System scope is currently not supported.");
        auto atomicImpl = trait::getAtomicImpl(acc[object::exec], scope);
        return internalCompute::Atomic::Op<T_Op, ALPAKA_TYPEOF(atomicImpl), T, T_Scope>::atomicOp(
            atomicImpl,
            addr,
            compare,
            value);
    }

    //! Executes an atomic add operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicAdd(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Add>(acc, addr, value, scope);
    }

    //! Executes an atomic sub operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicSub(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Sub>(acc, addr, value, scope);
    }

    //! Executes an atomic min operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicMin(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Min>(acc, addr, value, scope);
    }

    //! Executes an atomic max operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicMax(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Max>(acc, addr, value, scope);
    }

    //! Executes an atomic exchange operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicExch(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Exch>(acc, addr, value, scope);
    }

    //! Executes an atomic increment operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicInc(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Inc>(acc, addr, value, scope);
    }

    //! Executes an atomic decrement operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicDec(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Dec>(acc, addr, value, scope);
    }

    //! Executes an atomic and operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicAnd(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::And>(acc, addr, value, scope);
    }

    //! Executes an atomic or operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicOr(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Or>(acc, addr, value, scope);
    }

    //! Executes an atomic xor operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicXor(auto const& acc, T* const addr, T const& value, T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Xor>(acc, addr, value, scope);
    }

    //! Executes an atomic compare-and-swap operation.
    //!
    //! \tparam T The value type.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    template<typename T, typename T_Scope = scope::Device>
    constexpr auto atomicCas(
        auto const& acc,
        T* const addr,
        T const& compare,
        T const& value,
        T_Scope const scope = T_Scope()) -> T
    {
        return atomicOp<operation::Cas>(acc, addr, compare, value, scope);
    }

    namespace atomic
    {
        /** Defines the equivalent of an atomic invoke for user defined functors.
         *
         * This function can be specialized within the namespace of the user functor type and will be found via ADL.
         * Typically, this functor is used within the reduce and transformReduce algorithm.
         * The implementation must implement the functor equivalent atomic function.
         *
         * @param fn non-atomic user functor
         * @param inOut pointer to the values which is updated
         * @param args arguments normally forwarded to the user functor
         */
        ALPAKA_FN_ACC void alpakaAtomicInvoke(auto&& fn, concepts::Acc auto const& acc, auto* inOut, auto&&... args)
        {
            alpaka::unused(acc, inOut, args...);
            static_assert(
                sizeof(ALPAKA_TYPEOF(fn)) && false,
                "You must specialize alpakaAtomicInvoke() for your functor. Best place the overload in the namespace "
                "of the "
                "functor, it will be found by ADL.");
        }

        template<typename T>
        ALPAKA_FN_ACC void alpakaAtomicInvoke(std::plus<T>, concepts::Acc auto const& acc, auto* inOut, auto&&... args)
        {
            atomicAdd(acc, inOut, ALPAKA_FORWARD(args)...);
        }
    } // namespace atomic

} // namespace alpaka::onAcc
