/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onHost/algo/internal/transformReduce.hpp"

namespace alpaka::onHost
{
    /** accumulate the results into a scalar value.
     *
     * @param queue The queue to execute the transformation.
     * @param exec The executor to use for the kernel execution.
     * @param neutralElement The neutral element in respect to binaryReduceFn.
     * @param out MdSpan for the result. The value_type must be equal to neutralElement and the result of the binary
     * reduce functor type. The result is written to the first element of the output data.
     * @param binaryReduceFn Reduce binary functor, the functor operation must be transitive and commutative.
     *   The atomic operation atomic::alpakaAtomicInvoke(ReduceFnType, onAcc::concepts::Acc, auto* destination,auto
     * source) must be overloaded. The functor execution order is not specified. The functor should support Simd
     * packages, if not you can enforce the element wise execution by wrapping into ScalarFunc.
     * @param in The input data which should be reduced.
     *
     * @{
     */
    template<typename T_DataType, typename T_Device, alpaka::concepts::QueuePolicyList T_Policies>
    inline void reduce(
        Queue<T_Device, T_Policies> const& queue,
        alpaka::concepts::Executor auto const exec,
        T_DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto&& out,
        auto&& binaryReduceFn,
        alpaka::concepts::IDataSource auto&& in)
        requires(std::same_as<T_DataType, alpaka::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::transformReduce(
                queue,
                defaultExecutor(queue.getDevice()),
                neutralElement,
                ALPAKA_FORWARD(out),
                ALPAKA_FORWARD(binaryReduceFn),
                std::identity{},
                ALPAKA_FORWARD(in));
        }
        else
            internal::transformReduce(
                queue,
                exec,
                neutralElement,
                ALPAKA_FORWARD(out),
                ALPAKA_FORWARD(binaryReduceFn),
                std::identity{},
                ALPAKA_FORWARD(in));
    }

    /**
     * An available default executor will be selected automatically. The default executor is an executor with most
     * parallelism/performance.
     */
    template<typename T_DataType, typename T_Device, alpaka::concepts::QueuePolicyList T_Policies>
    inline void reduce(
        Queue<T_Device, T_Policies> const& queue,
        T_DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto&& out,
        auto&& binaryReduceFn,
        alpaka::concepts::IDataSource auto&& in)
        requires(std::same_as<T_DataType, alpaka::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        reduce(
            queue,
            defaultExecutor(queue.getDevice()),
            neutralElement,
            ALPAKA_FORWARD(out),
            ALPAKA_FORWARD(binaryReduceFn),
            ALPAKA_FORWARD(in));
    }

    /** @} */
} // namespace alpaka::onHost
