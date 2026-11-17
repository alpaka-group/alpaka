/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onAcc/internal/SimdConcurrent.hpp"
#include "alpaka/onAcc/internal/SimdTransformReduce.hpp"

#include <bit>
#include <cstdint>

namespace alpaka::onAcc
{
    /** Creates a functor operate on contiguous data concurrently.
     *
     * The class is automatically configured to use the best fitting SIMD width for the given data type and is able to
     * expose instruction level parallelism.
     *
     * @param T_WorkGroup participating thread description. More than one thread can have the same index within the
     * group. All worker with the same id will get the same index as result.
     * @param T_Traverse Policy to configure the method used to find the next valid index for a worker. @see namespace
     * traverse
     * @param T_IdxLayout Policy to define how index data will be mapped to worker threads. @see namespace layout
     */
    template<
        typename T_WorkGroup,
        concepts::IdxTraversing T_Traverse = traverse::Flat,
        concepts::IdxMapping T_IdxLayout = layout::Optimized>
    struct SimdAlgo
        : protected internal::SimdConcurrent<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>
        , protected internal::SimdTransformReduce<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>
    {
        constexpr SimdAlgo(
            T_WorkGroup const workGroup,
            T_Traverse traverse = T_Traverse{},
            T_IdxLayout idxLayout = T_IdxLayout{})
            : m_workGroup{workGroup}
        {
            alpaka::unused(traverse, idxLayout);
        }

        constexpr T_WorkGroup getWorkGroup() const
        {
            return m_workGroup;
        }

        constexpr T_Traverse getTraversePolicy() const
        {
            return T_Traverse{};
        }

        constexpr T_IdxLayout getIdxLayoutPolicy() const
        {
            return T_IdxLayout{};
        }

        /** execute the functor concurrently over the given data.
         *
         * @attention The number of elements to process is derived from the first MdSpan object.
         *            All other MdSpan objects must have at least the same number of elements.
         *            The optimal concurrency is also derived from the first MdSpan.
         *
         * @param func the functor to be executed
         * @param data0 the first data to be processed
         * @param dataN the remaining data to be processed
         *
         * @{
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            concurrent(acc, data0.getExtents(), ALPAKA_FORWARD(func), ALPAKA_FORWARD(data0), ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @param extents number of elements to process in each dimension
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            using ValueType = alpaka::GetValueType_t<ALPAKA_TYPEOF(data0)>;
            concurrent<
                alpaka::getNumElemPerThread<ValueType>(
                    ALPAKA_TYPEOF(acc.getApi()){},
                    ALPAKA_TYPEOF(acc.getDeviceKind()){})
                * sizeof(ValueType)>(
                acc,
                extents,
                ALPAKA_FORWARD(func),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /** @} */

        /** execute the functor concurrently over the given data.
         *
         * @attention The number of elements to process is derived from the first MdSpan object.
         *            All other MdSpan objects must have at least the same number of elements.
         *
         * @tparam T_maxConcurrencyInByte
         *    Maximum number of bytes to be used for concurrency.
         *    Concurrency bytes describe a virtual simd pack size which is not exceeded.
         *    Internally a best fitting SIMD width is calculated and instruction parallelism is exposed based on
         *    T_maxConcurrencyInByte.
         * @tparam T_MemAlignment alignment of the memory, if no alignments is given the alignment will be derived from
         * the MdSpan data descriptions
         * @param func the functor to be executed
         * @param data0 the first data to be processed
         * @param dataN the remaining data to be processed
         *
         * @{
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            concurrent<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                data0.getExtents(),
                ALPAKA_FORWARD(func),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @param extents number of elements to process in each dimension
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto&& func,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            ConcurrentAlgo::template concurrent<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                extents,
                ALPAKA_FORWARD(func),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /** @} */


        /** @brief transform the input data and reduce it to a single value
         *
         * @attention If no extent is given the number of elements to process is derived from the first MdSpan object.
         *            All other MdSpan objects must have at least the same number of elements.
         *
         * @param neutralElement the neutral element for the reduction operation
         * @param reduceFunc The binary reduction operation to be executed, e.g. std::plus. The functor should support
         * Simd packages.
         * @param transformFunc N-nary functor to be executed, values of all containers will be passed to the functor
         * as arguments. The functor should support Simd packages. If not you can enforce the element wise execution by
         * wrapping into
         * ScalarFunc. If you would like to support stencil executions wrapp fn into StencilFunc. StencilFunc
         * is getting all arguments as SimdPtr. If StencilFunc is used you should take care to not read outside of
         * valid memory ranges by using sub-views to your input and output data. Optionally a transformFn can have an
         * accelerator as first argument.
         * If the result of this functor is a structured value providing an overload to simdize the type
         * can improve the performance see alpaka::makeSimdized.
         * @param data0 the first data to be processed
         * @param dataN the remaining data to be processed
         * @return A single reduced value.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            return transformReduce(
                acc,
                data0.getExtents(),
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @copydoc transformReduce()
         * @param extents number of elements to process in each dimension
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            using ValueType = alpaka::GetValueType_t<ALPAKA_TYPEOF(data0)>;
            return transformReduce<
                alpaka::getNumElemPerThread<ValueType>(
                    ALPAKA_TYPEOF(acc.getApi()){},
                    ALPAKA_TYPEOF(acc.getDeviceKind()){})
                * sizeof(ValueType)>(
                acc,
                extents,
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @copydoc transformReduce()
         *
         * @tparam T_maxConcurrencyInByte
         *    Maximum number of bytes to be used for concurrency.
         *    Concurrency bytes describe a virtual simd pack size which is not exceeded.
         *    Internally a best fitting SIMD width is calculated and instruction parallelism is exposed based on
         *    T_maxConcurrencyInByte.
         * @tparam T_MemAlignment alignment of the memory, if no alignments is given the alignment will be derived from
         * the MdSpan data descriptions
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            return transformReduce<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                data0.getExtents(),
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

        /**
         * @copydoc transformReduce()
         *
         * @param extents number of elements to process in each dimension
         * @tparam T_maxConcurrencyInByte
         *    Maximum number of bytes to be used for concurrency.
         *    Concurrency bytes describe a virtual simd pack size which is not exceeded.
         *    Internally a best fitting SIMD width is calculated and instruction parallelism is exposed based on
         *    T_maxConcurrencyInByte.
         * @tparam T_MemAlignment alignment of the memory, if no alignments is given the alignment will be derived from
         * the MdSpan data descriptions
         */
        template<uint32_t T_maxConcurrencyInByte, alpaka::concepts::Alignment T_MemAlignment = AutoAligned>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr auto transformReduce(
            auto const& acc,
            alpaka::concepts::Vector auto extents,
            auto const& neutralElement,
            auto&& reduceFunc,
            auto&& transformFunc,
            alpaka::concepts::IDataSource auto&& data0,
            alpaka::concepts::IDataSource auto&&... dataN) const
        {
            return ReduceAlgo::template transformReduce<T_maxConcurrencyInByte, T_MemAlignment>(
                acc,
                extents,
                neutralElement,
                ALPAKA_FORWARD(reduceFunc),
                ALPAKA_FORWARD(transformFunc),
                ALPAKA_FORWARD(data0),
                ALPAKA_FORWARD(dataN)...);
        }

    private:
        using ConcurrentAlgo = internal::SimdConcurrent<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>;
        using ReduceAlgo = internal::SimdTransformReduce<SimdAlgo<T_WorkGroup, T_Traverse, T_IdxLayout>>;

        friend ConcurrentAlgo;
        friend ReduceAlgo;

        template<typename T_Type, uint32_t T_maxConcurrencyInByte, uint32_t T_cacheLineInByte>
        static constexpr auto calcSimdWidth()
        {
            constexpr uint32_t maxSimdBytes = std::min(T_cacheLineInByte, T_maxConcurrencyInByte);
            return alpaka::divExZero(maxSimdBytes, static_cast<uint32_t>(sizeof(T_Type)));
        }

        template<typename T_Type>
        struct SimdPackConfig
        {
            using value_type = T_Type;
            uint32_t simdWidth;
            uint32_t numSimdPacksPerFnCall;
        };

        /** Generate a SIMD config for the API and device kind.
         *
         * Produces an optimized SIMD configuration based on technical constraints.
         * The SIMD is set to a power of two.
         * If possible, the SIMD configuration is aligned to the cacheline size for the given device kind.
         *
         * @param maxConcurrencyInByte The upper limit in bytes a SIMD configuration must not exceed, except a single
         * value is larger. This parameter is used to control the register pressure.
         *
         * @return a configuration with the number of SIMD pack which should be used in parallel for a single
         * invocation. And the width of a single SIMD pack.
         */
        template<typename T_ValueType>
        [[nodiscard]] static consteval SimdPackConfig<T_ValueType> calcSimdPackConfig(
            alpaka::concepts::Api auto api,
            alpaka::concepts::DeviceKind auto deviceKind,
            uint32_t maxConcurrencyInByte)
        {
            constexpr uint32_t maxArchSimdWidth = getArchSimdWidth<T_ValueType>(api, deviceKind);
            constexpr uint32_t cachelineBytes = getCachelineSize(api, deviceKind);
            uint32_t simdWidth = maxArchSimdWidth;

            // Maximum SIMD width allowed by the byte concurrency budget.
            uint32_t maxWidthAllowed = maxConcurrencyInByte / sizeof(T_ValueType);

            // Clamp max hardware SIMD width and ensure at least 1.
            uint32_t clampedWidth = std::max(std::min(simdWidth, maxWidthAllowed), 1u);

            // Round down to the nearest power of two.
            simdWidth = std::bit_floor(clampedWidth);

            uint32_t const simdWidthInByte = simdWidth * sizeof(T_ValueType);

            // Number of SIMD packs that fit into the concurrency budget.
            uint32_t const numSimdPacksToUtilizeConcurrency = alpaka::divExZero(maxConcurrencyInByte, simdWidthInByte);

            // Number of SIMD packs required to cover one cache line
            uint32_t const numSimdPacksPerCacheLine = alpaka::divExZero(cachelineBytes, simdWidthInByte);

            // Prefer the largest cache-line multiple that fits into the budget.
            uint32_t numSimdPacksPerFnCall = numSimdPacksToUtilizeConcurrency;
            if(numSimdPacksToUtilizeConcurrency >= numSimdPacksPerCacheLine)
            {
                uint32_t const cachelineMultiple
                    = (numSimdPacksToUtilizeConcurrency / numSimdPacksPerCacheLine) * numSimdPacksPerCacheLine;
                numSimdPacksPerFnCall = std::max(cachelineMultiple, 1u);
            }

            return {simdWidth, numSimdPacksPerFnCall};
        }

        T_WorkGroup m_workGroup;
    };
} // namespace alpaka::onAcc
