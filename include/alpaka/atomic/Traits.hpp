/* Copyright 2025 Benjamin Worpitz, René Widera, Bernhard Manfred Gruber, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/atomic/Op.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/core/Positioning.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

#include <type_traits>

namespace alpaka
{
    struct ConceptAtomicGrids
    {
    };

    struct ConceptAtomicBlocks
    {
    };

    struct ConceptAtomicThreads
    {
    };

    namespace detail
    {
        template<typename THierarchy>
        struct AtomicHierarchyConceptType;

        template<>
        struct AtomicHierarchyConceptType<hierarchy::Threads>
        {
            using type = ConceptAtomicThreads;
        };

        template<>
        struct AtomicHierarchyConceptType<hierarchy::Blocks>
        {
            using type = ConceptAtomicBlocks;
        };

        template<>
        struct AtomicHierarchyConceptType<hierarchy::Grids>
        {
            using type = ConceptAtomicGrids;
        };
    } // namespace detail

    template<typename THierarchy>
    using AtomicHierarchyConcept = typename detail::AtomicHierarchyConceptType<THierarchy>::type;

    //! The atomic operation trait.
    namespace trait
    {
        //! The atomic operation trait.
        template<
            typename TOp,
            typename TAtomic,
            typename T,
            MemoryOrder TMemOrder,
            typename THierarchy,
            typename TSfinae = void>
        struct AtomicOp;
    } // namespace trait

    //! Executes the given operation atomically.
    //!
    //! \tparam TOp The operation type.
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \tparam TMemOrder The memory order type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    //! \param order The memory order.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOp, typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOp(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& = THierarchy()) -> T
    {
        using ImplementationBase = typename interface::ImplementationBase<AtomicHierarchyConcept<THierarchy>, TAtomic>;
        return trait::AtomicOp<TOp, ImplementationBase, T, TMemOrder, THierarchy>::atomicOp(
            atomic,
            addr,
            value,
            order);
    }

    //! Executes the given operation atomically (compare-and-swap variant).
    //!
    //! \tparam TOp The operation type.
    //! \tparam TAtomic The atomic implementation type.
    //! \tparam T The value type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOp, typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOp(
        TAtomic const& atomic,
        T* const addr,
        T const& compare,
        T const& value,
        TMemOrder order,
        THierarchy = THierarchy()) -> T
    {
        using ImplementationBase = typename interface::ImplementationBase<AtomicHierarchyConcept<THierarchy>, TAtomic>;
        return trait::AtomicOp<TOp, ImplementationBase, T, TMemOrder, THierarchy>::atomicOp(
            atomic,
            addr,
            compare,
            value,
            order);
    }

    //! Executes the given operation atomically (with default relaxed memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOp, typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOp(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy hier = THierarchy()) -> T
    {
        return atomicOp<TOp>(atomic, addr, value, mem_order::relaxed, hier);
    }

    //! Executes the given operation atomically (compare-and-swap variant with default relaxed memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TOp, typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOp(
        TAtomic const& atomic,
        T* const addr,
        T const& compare,
        T const& value,
        THierarchy hier = THierarchy()) -> T
    {
        return atomicOp<TOp>(atomic, addr, compare, value, mem_order::relaxed, hier);
    }

    //! Executes an atomic add operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \tparam TMemOrder The memory order type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicAdd(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAdd>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic add operation (with default memory order).
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicAdd(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAdd>(atomic, addr, value, hier);
    }

    //! Executes an atomic sub operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicSub(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicSub>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic sub operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicSub(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicSub>(atomic, addr, value, hier);
    }

    //! Executes an atomic min operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicMin(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMin>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic min operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicMin(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMin>(atomic, addr, value, hier);
    }

    //! Executes an atomic max operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicMax(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMax>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic max operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicMax(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicMax>(atomic, addr, value, hier);
    }

    //! Executes an atomic exchange operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicExch(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicExch>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic exchange operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicExch(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicExch>(atomic, addr, value, hier);
    }

    //! Executes an atomic increment operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicInc(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicInc>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic increment operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicInc(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicInc>(atomic, addr, value, hier);
    }

    //! Executes an atomic increment operation, using the numeric limit as ceiling.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    //! NOTE: The limit value is deduced as std::numeric_limits<T>::max()
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicInc(
        TAtomic const& atomic,
        T* const addr,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        T const value = std::numeric_limits<T>::max();
        return atomicOp<AtomicInc>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic increment operation, using the numeric limit as ceiling (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicInc(TAtomic const& atomic, T* const addr, THierarchy const& hier = THierarchy()) -> T
    {
        T const value = std::numeric_limits<T>::max();
        return atomicOp<AtomicInc>(atomic, addr, value, hier);
    }

    //! Executes an atomic decrement operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicDec(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicDec>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic decrement operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicDec(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicDec>(atomic, addr, value, hier);
    }

    //! Executes an atomic decrement operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    //! NOTE: The limit value is deduced as std::numeric_limits<T>::max()
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicDec(
        TAtomic const& atomic,
        T* const addr,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        T const value = std::numeric_limits<T>::max();
        return atomicOp<AtomicDec>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic decrement operation, using the numeric limit as ceiling (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicDec(TAtomic const& atomic, T* const addr, THierarchy const& hier = THierarchy()) -> T
    {
        T const value = std::numeric_limits<T>::max();
        return atomicOp<AtomicDec>(atomic, addr, value, hier);
    }

    //! Executes an atomic and operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicAnd(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAnd>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic and operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicAnd(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicAnd>(atomic, addr, value, hier);
    }

    //! Executes an atomic or operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOr(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicOr>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic or operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOr(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicOr>(atomic, addr, value, hier);
    }

    //! Executes an atomic xor operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicXor(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicXor>(atomic, addr, value, order, hier);
    }

    //! Executes an atomic xor operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicXor(
        TAtomic const& atomic,
        T* const addr,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicXor>(atomic, addr, value, hier);
    }

    //! Executes an atomic compare-and-swap operation.
    //!
    //! \tparam TAtomic The atomic implementation type.
    //! \tparam T The value type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    //! \param order The memory order.
    //! \param hier The hierarchy level.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, MemoryOrder TMemOrder, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicCas(
        TAtomic const& atomic,
        T* const addr,
        T const& compare,
        T const& value,
        TMemOrder order,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicCas>(atomic, addr, compare, value, order, hier);
    }

    //! Executes an atomic compare-and-swap operation (with default memory order).
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAtomic, typename T, typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicCas(
        TAtomic const& atomic,
        T* const addr,
        T const& compare,
        T const& value,
        THierarchy const& hier = THierarchy()) -> T
    {
        return atomicOp<AtomicCas>(atomic, addr, compare, value, hier);
    }


} // namespace alpaka
