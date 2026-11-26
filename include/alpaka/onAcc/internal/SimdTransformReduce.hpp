/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/SimdPtr.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/functor.hpp"
#include "alpaka/mem/concepts/IDataSource.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/WorkerGroup.hpp"
#include "alpaka/onAcc/interface.hpp"
#include "alpaka/simd/simdized.hpp"

#include <cstdint>
#include <new>

namespace alpaka::onAcc::internal
{

    /** concurrent reduce implementation */
    template<typename T_Derived>
    struct SimdTransformReduce
    {
    private:
        // only the derived class T_Derived may create the base
        constexpr SimdTransformReduce() = default;
        friend T_Derived;

    protected:
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            concepts::Acc auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            auto numElements = typename ALPAKA_TYPEOF(extents)::UniVec{extents};
            using ValueType = alpaka::GetValueType_t<ALPAKA_TYPEOF(data0)>;
            decltype(auto) transformFunc = wrapTransformFunc(ALPAKA_FORWARD(func));

            constexpr auto simdCfg = T_Derived::template calcSimdPackConfig<ValueType>(
                ALPAKA_TYPEOF(acc.getApi()){},
                ALPAKA_TYPEOF(acc.getDeviceKind()){},
                T_maxConcurrencyInByte);

            constexpr uint32_t simdWidth = simdCfg.simdWidth;

            if constexpr(simdWidth != 1u)
            {
                constexpr uint32_t numSimdPerFnCall = simdCfg.numSimdPacksPerFnCall;
                return reduceSimdPackExecution<simdWidth, numSimdPerFnCall, T_MemAlignment>(
                    acc,
                    numElements,
                    neutralElement,
                    ALPAKA_FORWARD(reduceFunc),
                    transformFunc,
                    ALPAKA_FORWARD(data0),
                    ALPAKA_FORWARD(dataN)...);
            }

            auto const workGroup = asDerived().getWorkGroup();
            // execute the algorithm with SIMD width one
            auto traverse = onAcc::makeIdxMap(
                acc,
                workGroup,
                IdxRange{numElements},
                asDerived().getTraversePolicy(),
                asDerived().getIdxLayoutPolicy());

            using SimdOneReturnType = ALPAKA_TYPEOF(makeSimdized<1u>(neutralElement));
            SimdOneReturnType simdizedReducedValue = makeSimdized<1u>(neutralElement);

            for(auto idx : traverse)
            {
                simdizedReducedValue = reduceFunc(
                    simdizedReducedValue,
                    callFunctor(
                        acc,
                        transformFunc,
                        SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                        SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...));
            }

            auto result = neutralElement;
            alpakaSimdizedInvoke(
                [](auto& lhs, alpaka::concepts::Simd auto const& rhs) { lhs = rhs[0]; },
                result,
                simdizedReducedValue);
            return result;
        }

    private:
        template<uint32_t... T_idx>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static constexpr auto loadAncExecuteScalarOp(
            std::integer_sequence<uint32_t, T_idx...>,
            auto&& op,
            auto const& acc,
            auto&& func,
            auto&&... data)
        {
            return Simd{op(CVec<uint32_t, T_idx>{}, acc, func, data...)...};
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC static constexpr decltype(auto) wrapTransformFunc(auto&& transformFunc)
        {
            if constexpr(isSpecializationOf_v<ALPAKA_TYPEOF(transformFunc), StencilFunc>)
            {
                return ALPAKA_FORWARD(transformFunc);
            }
            else if constexpr(isSpecializationOf_v<ALPAKA_TYPEOF(transformFunc), ScalarFunc>)
            {
                return [transformFunc = ALPAKA_FORWARD(transformFunc)](
                           onAcc::concepts::Acc auto const& acc,
                           alpaka::concepts::SimdPtr auto&& inPtr0,
                           alpaka::concepts::SimdPtr auto const&... inPtr) constexpr
                {
                    return loadAncExecuteScalarOp(
                        std::make_integer_sequence<uint32_t, ALPAKA_TYPEOF(inPtr0)::width()>{},
                        [](alpaka::concepts::CVector auto idx,
                           auto const& acc,
                           auto&& func,
                           alpaka::concepts::Simd auto const&... data) constexpr
                        { return callFunctor(acc, func, data[idx.x()]...); },
                        acc,
                        transformFunc,
                        inPtr0.load(),
                        inPtr.load()...);
                };
            }
            else
            {
                return [transformFunc = ALPAKA_FORWARD(transformFunc)](
                           onAcc::concepts::Acc auto const& acc,
                           alpaka::concepts::SimdPtr auto&&... inPtr) constexpr
                { return callFunctor(acc, transformFunc, inPtr.load()...); };
            }
        }

        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width>
        ALPAKA_FN_INLINE static constexpr auto executeDoTransform(
            concepts::Acc auto const& acc,
            auto const& dataIdx,
            auto&& func,
            alpaka::concepts::IDataSource auto&&... data)
        {
            return callFunctor(acc, func, SimdPtr{data, dataIdx, T_MemAlignment{}, CVec<uint32_t, T_width>{}}...);
        }

        /** advance the iterator T_repeat times
         *
         * We do not check if the iterator points to a valid element, the caller must ensure that we can safely
         * advance the iterator T_repeat time without jumping over iter.end().
         *
         * @tparam T_repeat Number of time sthe iterator should be advanced.
         * @return Tuple with T_repeat times iterators.
         */
        template<uint32_t... T_repeat>
        ALPAKA_FN_INLINE static constexpr auto makeAdvanceIterators(
            auto& iter,
            std::integer_sequence<uint32_t, T_repeat...>)
        {
            // The ternary operator is used to allow using the folding expression on iter.
            // NOLINTNEXTLINE(bugprone-branch-clone)
            return std::make_tuple(*(T_repeat + 1 != 0u ? iter++ : iter++)...);
        }

        /** Calls the transform functor T_repeat times and reduces the results with the given reduce function.
         *
         * The calls to the functor are independent and compile time unrolled to support instruction parallelism.
         * In contrast to executeReduceInto() the register footprint is larger because T_repeat temporary results will
         * be holt. This allows the compiler to use instruction level parallelism. Call this function if result of
         * reduceFunc is a SIMD pack.
         *
         * @param iter the caller must ensure tha the interator can be increased T_repeat times without jumping over
         * iter.end()
         * @return a single simdized pack
         */
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width, uint32_t... T_repeat>
        ALPAKA_FN_INLINE static constexpr auto executeReduce(
            concepts::Acc auto const& acc,
            auto& iter,
            std::integer_sequence<uint32_t, T_repeat...>,
            auto&& reduceFunc,
            auto&& func,
            alpaka::concepts::IDataSource auto&&... data)
        {
            auto ids = makeAdvanceIterators(iter, std::integer_sequence<uint32_t, T_repeat...>{});
            return std::apply(
                [&](auto const&... dataIdx) constexpr
                {
                    /* It is not possible to create a Simd{Simd} due to constructor issues. Therefore we need to define
                     * the type for the result explicit.
                     */
                    using ComponentType = ALPAKA_TYPEOF(
                        executeDoTransform<T_MemAlignment, T_width>(
                            acc,
                            std::get<0>(std::make_tuple(dataIdx...)),
                            func,
                            data...));
                    auto results = Simd<ComponentType, std::tuple_size_v<ALPAKA_TYPEOF(ids)>>{
                        executeDoTransform<T_MemAlignment, T_width>(acc, dataIdx, func, data...)...};

                    return results.reduce(reduceFunc);
                },
                ids);
        }

        /** Reduce simdized packs into a single simdized pack with the given reduce function.
         *
         * In contrast to executeReduce() the register footprint is lower because all intermediate results are directly
         * reduced into the result variable. Call this function if the type of result is a simdized pack is not a SIMD
         * pack.
         *
         * @param result The results of reduceFn with the result of transformFn will be reduced into this simdized
         * pack.
         */
        template<alpaka::concepts::Alignment T_MemAlignment, uint32_t T_width, uint32_t... T_repeat>
        ALPAKA_FN_INLINE static constexpr void executeReduceInto(
            concepts::Acc auto const& acc,
            auto& iter,
            std::integer_sequence<uint32_t, T_repeat...>,
            auto& result,
            auto&& reduceFn,
            auto&& transformFn,
            alpaka::concepts::IDataSource auto&&... data)
        {
            auto ids = makeAdvanceIterators(iter, std::integer_sequence<uint32_t, T_repeat...>{});
            std::apply(
                [&](auto const&... dataIdx) constexpr
                {
                    ((result = reduceFn(
                          result,
                          executeDoTransform<T_MemAlignment, T_width>(acc, dataIdx, transformFn, data...))),
                     ...);
                },
                ids);
        }

        /** Reduce T_numSimdPerFnCall simdized packs
         *
         */
        template<uint32_t T_simdWidth, uint32_t T_numSimdPerFnCall, alpaka::concepts::Alignment T_MemAlignment>
        ALPAKA_FN_INLINE static constexpr void reduceNextSimdized(
            auto const& acc,
            auto& iter,
            auto& tmpReturn,
            auto&& reduceFn,
            auto&& transformFn,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN)
        {
            if constexpr(alpaka::concepts::Simd<std::remove_cvref_t<decltype(tmpReturn)>>)
            {
                tmpReturn = reduceFn(
                    tmpReturn,
                    executeReduce<T_MemAlignment, T_simdWidth>(
                        acc,
                        iter,
                        std::make_integer_sequence<uint32_t, T_numSimdPerFnCall>{},
                        reduceFn,
                        transformFn,
                        data0,
                        dataN...));
            }
            else
            {
                executeReduceInto<T_MemAlignment, T_simdWidth>(
                    acc,
                    iter,
                    std::make_integer_sequence<uint32_t, T_numSimdPerFnCall>{},
                    tmpReturn,
                    reduceFn,
                    transformFn,
                    data0,
                    dataN...);
            }
        }

        template<onAcc::concepts::Acc T_Acc, typename T_ReduceOp>
        class ScalarReducer
        {
            // using a const reference here is fine because we control the lifetime
            T_Acc const& m_acc;
            T_ReduceOp const& m_reduceOp;

        public:
            constexpr ScalarReducer(T_Acc const& acc, auto&& func) : m_acc(acc), m_reduceOp{ALPAKA_FORWARD(func)}
            {
            }

            constexpr auto operator()(auto&& a, auto&& b) const
                requires(alpaka::concepts::Simd<ALPAKA_TYPEOF(a)> && alpaka::concepts::Simd<ALPAKA_TYPEOF(b)>)
            {
                return loadAncExecuteScalarOp(
                    std::make_integer_sequence<uint32_t, ALPAKA_TYPEOF(a)::width()>{},
                    [this](
                        alpaka::concepts::CVector auto idx,
                        concepts::Acc auto const& acc,
                        auto&& func,
                        auto const&... data) constexpr
                    {
                        /* const& for data is used instead of && to enforce const evaluation of the operator[]
                         * std simd operator[] is returning a smart reference which is avoided if data is const
                         */
                        alpaka::unused(acc, func);
                        // recursively call until no Simd type is the result
                        return this->operator()(data[idx.x()]...);
                    },
                    m_acc,
                    ALPAKA_FORWARD(m_reduceOp),
                    ALPAKA_FORWARD(a),
                    ALPAKA_FORWARD(b));
            }

            constexpr auto operator()(auto&& a, auto&& b) const
                requires(!alpaka::concepts::Simd<ALPAKA_TYPEOF(a)> && !alpaka::concepts::Simd<ALPAKA_TYPEOF(b)>)
            {
                return m_reduceOp(ALPAKA_FORWARD(a), ALPAKA_FORWARD(b));
            }

        private:
            template<uint32_t... T_idx>
            ALPAKA_FN_INLINE ALPAKA_FN_ACC static constexpr auto loadAncExecuteScalarOp(
                std::integer_sequence<uint32_t, T_idx...>,
                auto&& op,
                auto const& acc,
                auto&& func,
                auto&&... data)
            {
                return Simd{op(CVec<uint32_t, T_idx>{}, acc, func, data...)...};
            }
        };

        /** Get the reducer functor
         *
         * @return wrapped functor in case the input is @see ScalarFunc else the identity
         */
        ALPAKA_FN_INLINE constexpr auto getReducer(onAcc::concepts::Acc auto const&, auto&& reduceOp) const
            requires(!isSpecializationOf_v<ALPAKA_TYPEOF(reduceOp), ScalarFunc>)
        {
            return reduceOp;
        }

        ALPAKA_FN_INLINE constexpr auto getReducer(onAcc::concepts::Acc auto const& acc, auto&& reduceOp) const
            requires(isSpecializationOf_v<ALPAKA_TYPEOF(reduceOp), ScalarFunc>)
        {
            return ScalarReducer<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(reduceOp)>{acc, reduceOp};
        }

        constexpr auto const& asDerived() const
        {
            return static_cast<T_Derived const&>(*this);
        }

        template<uint32_t T_simdWidth, uint32_t T_numSimdPerFnCall, alpaka::concepts::Alignment T_MemAlignment>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto reduceSimdPackExecution(
            auto const& acc,
            alpaka::concepts::Vector auto numElements,
            auto const& neutralElement,
            auto&& userReduceFunc,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            auto reduceFunc = getReducer(acc, userReduceFunc);

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
            auto simdIdxContainer = onAcc::makeIdxMap(
                acc,
                workGroup,
                IdxRange{IdxType::fill(0), domainSize, stride},
                asDerived().getTraversePolicy(),
                asDerived().getIdxLayoutPolicy());

            using SimdReturn = ALPAKA_TYPEOF(makeSimdized<T_simdWidth>(neutralElement));
            SimdReturn simdizedReducedValue = makeSimdized<T_simdWidth>(neutralElement);

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

                    // iterate over the fast-moving dimension only
                    auto simdIdxContainerFastDim = onAcc::makeIdxMap(
                        acc,
                        wInner,
                        IdxRange{rowIdx, domainSize, stride},
                        asDerived().getTraversePolicy(),
                        asDerived().getIdxLayoutPolicy())[CVec<uint32_t, ALPAKA_TYPEOF(domainSize)::dim() - 1u>{}];

                    for(auto iter = simdIdxContainerFastDim.begin(); iter != simdIdxContainerFastDim.end();)
                    {
                        reduceNextSimdized<T_simdWidth, T_numSimdPerFnCall, T_MemAlignment>(
                            acc,
                            iter,
                            simdizedReducedValue,
                            reduceFunc,
                            func,
                            data0,
                            dataN...);
                    }
                }
            }
            else
            {
                for(auto iter = simdIdxContainer.begin(); iter != simdIdxContainer.end();)
                {
                    reduceNextSimdized<T_simdWidth, T_numSimdPerFnCall, T_MemAlignment>(
                        acc,
                        iter,
                        simdizedReducedValue,
                        reduceFunc,
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
                auto transformResult = callFunctor(
                    acc,
                    func,
                    SimdPtr{data0, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}},
                    SimdPtr{dataN, idx, T_MemAlignment{}, CVec<uint32_t, 1u>{}}...);

                alpakaSimdizedInvoke(
                    [reduceFunc](auto& lhs, alpaka::concepts::Simd auto const& rhs)
                    {
                        // std simd non-const operator[] is returning a smart reference, therefore we need
                        // std::as_const to enforce returning a copy of the value.
                        lhs[0] = reduceFunc(std::as_const(lhs)[0], rhs[0]);
                    },
                    simdizedReducedValue,
                    transformResult);
            }

            ALPAKA_TYPEOF(neutralElement) result;
            alpakaSimdizedInvoke(
                [reduceFunc](auto& lhs, alpaka::concepts::Simd auto const& rhs) { lhs = rhs.reduce(reduceFunc); },
                result,
                simdizedReducedValue);
            return result;
        }
    };
} // namespace alpaka::onAcc::internal
