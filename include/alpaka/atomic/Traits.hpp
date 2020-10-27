/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/atomic/Op.hpp>

#include <type_traits>

namespace alpaka
{
    struct ConceptAtomicGrids{};
    struct ConceptAtomicBlocks{};
    struct ConceptAtomicThreads{};

    namespace detail
    {
        template<
            typename THierarchy
        >
        struct AtomicHierarchyConceptType;

        template<>
        struct AtomicHierarchyConceptType<
            hierarchy::Threads>
        {
            using type = ConceptAtomicThreads;
        };

        template<>
        struct AtomicHierarchyConceptType<
            hierarchy::Blocks>
        {
            using type = ConceptAtomicBlocks;
        };

        template<>
        struct AtomicHierarchyConceptType<
            hierarchy::Grids>
        {
            using type = ConceptAtomicGrids;
        };
    }

    template<
        typename THierarchy
    >
    using AtomicHierarchyConcept = typename detail::AtomicHierarchyConceptType<THierarchy>::type;

    //-----------------------------------------------------------------------------
    //! The atomic operation traits.
    namespace traits
    {
        //#############################################################################
        //! The atomic operation trait.
        template<
            typename TOp,
            typename TAtomic,
            typename T,
            typename THierarchy,
            typename TSfinae = void>
        struct AtomicOp;
    }

    //-----------------------------------------------------------------------------
    //! Executes the given operation atomically.
    //!
    //! \tparam TOp The operation type.
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TOp,
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOp(
        TAtomic const & atomic,
        T * const addr,
        T const & value,
        THierarchy const & = THierarchy())
    -> T
    {
        using ImplementationBase = typename concepts::ImplementationBase<AtomicHierarchyConcept<THierarchy>, TAtomic>;
        return
            traits::AtomicOp<
                TOp,
                ImplementationBase,
                T,
                THierarchy>
            ::atomicOp(
                atomic,
                addr,
                value);
    }

    //-----------------------------------------------------------------------------
    //! Executes the given operation atomically.
    //!
    //! \tparam TOp The operation type.
    //! \tparam TAtomic The atomic implementation type.
    //! \tparam T The value type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TOp,
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
    ALPAKA_FN_HOST_ACC auto atomicOp(
        TAtomic const & atomic,
        T * const addr,
        T const & compare,
        T const & value,
        THierarchy const & = THierarchy())
    -> T
    {
        using ImplementationBase = typename concepts::ImplementationBase<AtomicHierarchyConcept<THierarchy>, TAtomic>;
        return
            traits::AtomicOp<
                TOp,
                ImplementationBase,
                T,
                THierarchy>
            ::atomicOp(
                atomic,
                addr,
                compare,
                value);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic add operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicAdd(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Add>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic sub operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicSub(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Sub>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic min operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicMin(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Min>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic max operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicMax(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Max>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic exchange operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicExch(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Exch>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic increment operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicInc(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Inc>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic decrement operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicDec(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Dec>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic and operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicAnd(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::And>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic or operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicOr(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Or>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic xor operation.
    //!
    //! \tparam T The value type.
    //! \tparam TAtomic The atomic implementation type.
    //! \param addr The value to change atomically.
    //! \param value The value used in the atomic operation.
    //! \param atomic The atomic implementation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicXor(
            TAtomic const& atomic,
            T* const addr,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Xor>(atomic, addr, value, hier);
    }

    //-----------------------------------------------------------------------------
    //! Executes an atomic compare-and-swap operation.
    //!
    //! \tparam TAtomic The atomic implementation type.
    //! \tparam T The value type.
    //! \param atomic The atomic implementation.
    //! \param addr The value to change atomically.
    //! \param compare The comparison value used in the atomic operation.
    //! \param value The value used in the atomic operation.
    ALPAKA_NO_HOST_ACC_WARNING
        template<
        typename TAtomic,
        typename T,
        typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicCas(
            TAtomic const& atomic,
            T* const addr,
            T const& compare,
            T const& value,
            THierarchy const& hier = THierarchy())
        -> T
    {
        return atomicOp<op::Cas>(atomic, addr, compare, value, hier);
    }
}
