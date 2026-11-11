/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/SimdPtr.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/concepts/IDataSource.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onAcc/WorkerGroup.hpp"
#include "alpaka/onAcc/interface.hpp"

#include <cstdint>
#include <new>

namespace alpaka::onAcc::internal
{
    /** concurrent foreach implementation */
    template<typename T_Derived>
    struct SimdConcurrent
    {
    private:
        // only the derived class T_Derived may create the base
        constexpr SimdConcurrent() = default;
        friend T_Derived;

    protected:
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            auto numElements = typename ALPAKA_TYPEOF(extents)::UniVec{extents};
            using ValueType = alpaka::GetValueType_t<ALPAKA_TYPEOF(data0)>;

            constexpr auto simdCfg = T_Derived::template calcSimdPackConfig<ValueType>(
                ALPAKA_TYPEOF(acc.getApi()){},
                ALPAKA_TYPEOF(acc.getDeviceKind()){},
                T_maxConcurrencyInByte);

            constexpr uint32_t simdWidth = simdCfg.simdWidth;

            if constexpr(simdWidth != 1u)
            {
                constexpr uint32_t numSimdPerFnCall = simdCfg.numSimdPacksPerFnCall;
                concurrentSimdPackExecution<simdWidth, numSimdPerFnCall, T_MemAlignment>(
                    acc,
                    numElements,
                    ALPAKA_FORWARD(func),
                    ALPAKA_FORWARD(data0),
                    ALPAKA_FORWARD(dataN)...);
            }
            else
            {
                // execute the algorithm with SIMD width one
                for(auto idx : onAcc::makeIdxMap(
                        acc,
                        asDerived().getWorkGroup(),
                        IdxRange{numElements},
                        asDerived().getTraversePolicy(),
                        asDerived().getIdxLayoutPolicy()))
                {
                    func(
                        acc,
                        SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                        SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...);
                }
            }
        }

    private:
        constexpr auto const& asDerived() const
        {
            return static_cast<T_Derived const&>(*this);
        }

        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width>
        ALPAKA_FN_INLINE static constexpr void executeDo(
            auto const& acc,
            auto const& dataIdx,
            auto&& func,
            alpaka::concepts::IDataSource auto&&... data)
        {
            func(acc, SimdPtr{ALPAKA_FORWARD(data), dataIdx, T_MemAlignment{}, CVec<uint32_t, T_width>{}}...);
        }

        /** calls the functor and forward the data T_repeat times
         *
         * The calls to the functor are independent and compile time unrolled to support instruction parallelism.
         *
         * @param iter the caller must ensure tha the interator can be increased T_repeat times without jumping over
         * iter.end()
         */
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width, uint32_t... T_repeat>
        ALPAKA_FN_INLINE static constexpr void execute(
            auto const& acc,
            auto& iter,
            std::integer_sequence<uint32_t, T_repeat...>,
            auto&& func,
            alpaka::concepts::IDataSource auto&&... data)
        {
            /* We do not check if the iterator points to a valid element, the caller must ensure that we can safely
             * increase the iterator without jumping over iter.end().
             *
             * The ternary operator is used to allow using the folding expression on iter.
             */
            // NOLINTNEXTLINE(bugprone-branch-clone)
            auto ids = std::make_tuple(*(T_repeat + 1 != 0u ? iter++ : iter++)...);
            std::apply(
                [&](auto const&... dataIdx) constexpr
                { (executeDo<T_MemAlignment, T_width>(acc, dataIdx, func, data...), ...); },
                ids);
        }

        template<uint32_t T_simdWidth, uint32_t T_numSimdPerFnCall, alpaka::concepts::Alignment T_MemAlignment>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto concurrentSimdPackExecution(
            auto const& acc,
            alpaka::concepts::Vector auto numElements,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            auto const workGroup = asDerived().getWorkGroup();

            // we SIMDfy only over the fast moving dimension (columns of memory)
            auto const wSize = workGroup.getThreadCount(acc).back();

            /* Number of data elements processed per functor call. */
            auto const numElementsPerFnCall = T_simdWidth * T_numSimdPerFnCall;
            /** To avoid an overflow in the index range we divide first by the number of elements per
             * function call and then by the number of workers.
             */
            auto const numSimdPackLoops = numElements.back() / numElementsPerFnCall / wSize;

            // number of elements to jump over to start the remainder loop
            auto const remainderBegin = numSimdPackLoops * numElementsPerFnCall * wSize;

            // we SIMDfy only over the fast moving dimension (columns of memory)
            auto domainSize = numElements.rAssign(remainderBegin);
            auto stride = ALPAKA_TYPEOF(numElements)::fill(1).rAssign(T_simdWidth);
            using IdxType = ALPAKA_TYPEOF(numElements);

            if constexpr(
                domainSize.dim() > 1u
                && std::is_same_v<ALPAKA_TYPEOF(asDerived().getTraversePolicy()), traverse::Flat>)
            {
                /* For cases where we traverse with the flat policy, we cannot assume that we can blindly increase the
                 * iterator later N times. This could happen in cases where we have enough concurrency. We evaluate for
                 * SIMD operations only the fast moving dimension but with the flat policy flattening the worker group
                 * and use all workers on a linear domain. The loop must therefore be split into iterating over all
                 * slow dimensions and an inner loop iterating over the fast moving dimension. For this we need to
                 * build our own groups out of the user-provided workgroup.
                 */
                // build a worker group with slow-moving dimension threads for the outer loop
                using IndexType = typename IdxType::value_type;
                auto wIdx = workGroup.getThreadIdx(acc).rAssign(IndexType{0});
                auto wSize = workGroup.getThreadCount(acc).rAssign(IndexType{1});
                auto domSize = domainSize.rAssign(IndexType{1});

                auto wOuter = WorkerGroup{wIdx, wSize};

                for(auto rowIdx : onAcc::makeIdxMap(
                        acc,
                        wOuter,
                        IdxRange{domSize},
                        asDerived().getTraversePolicy(),
                        asDerived().getIdxLayoutPolicy()))
                {
                    // build a worker group with fast-moving dimension threads for the inner loop
                    auto wIdxInner = ALPAKA_TYPEOF(domainSize)::fill(0).rAssign(workGroup.getThreadIdx(acc).back());
                    auto wSizeInner = ALPAKA_TYPEOF(domainSize)::fill(1).rAssign(workGroup.getThreadCount(acc).back());
                    auto wInner = WorkerGroup{wIdxInner, wSizeInner};

                    // iterate over the fast-moving dimension
                    auto simdIdxContainer = onAcc::makeIdxMap(
                        acc,
                        wInner,
                        IdxRange{rowIdx, domainSize, stride},
                        asDerived().getTraversePolicy(),
                        asDerived().getIdxLayoutPolicy())[CVec<uint32_t, ALPAKA_TYPEOF(domainSize)::dim() - 1u>{}];

                    for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
                    {
                        execute<T_MemAlignment, T_simdWidth>(
                            acc,
                            iter,
                            std::make_integer_sequence<uint32_t, T_numSimdPerFnCall>{},
                            func,
                            data0,
                            dataN...);
                    }
                }
            }
            else
            {
                auto simdIdxContainer = onAcc::makeIdxMap(
                    acc,
                    workGroup,
                    IdxRange{IdxType::fill(0), domainSize, stride},
                    asDerived().getTraversePolicy(),
                    asDerived().getIdxLayoutPolicy());

                for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
                {
                    execute<T_MemAlignment, T_simdWidth>(
                        acc,
                        iter,
                        std::make_integer_sequence<uint32_t, T_numSimdPerFnCall>{},
                        func,
                        data0,
                        dataN...);
                }
            }

            ALPAKA_TYPEOF(numElements) remainderDomainSize = numElements.fill(0).rAssign(remainderBegin);

            for(auto idx : onAcc::makeIdxMap(
                    acc,
                    workGroup,
                    IdxRange{remainderDomainSize, numElements},
                    asDerived().getTraversePolicy(),
                    asDerived().getIdxLayoutPolicy()))
            {
                func(
                    acc,
                    SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                    SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...);
            }
        }
    };
} // namespace alpaka::onAcc::internal
