/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/mem/concepts/IDataStorage.hpp"
#include "alpaka/onHost/algo/internal/transformReduce.hpp"

namespace alpaka::onHost
{
    /** Transform the input data with the given function and accumulate the results into a scalar value.
     *
     * transformFn can be a lambda function if all arguments are specialized. This fully specialized functor must
     * mostly wrapped by @see ScalarFunc. Generic lambdas are for some backends e.g. CUDA/HIP not supported. A lambda
     * must be of the following form and should capture arguments only by copy.
     *
     * @code{.cpp}
     *   [] ALPAKA_FN_ACC(){};
     * @endcode
     *
     * @param queue The queue to execute the transformation.
     * @param exec The executor to use for the kernel execution.
     * @param neutralElement The neutral element in respect to binaryReduceFn.
     * @param out MdSpan for the result. The value_type must be equal to neutralElement and the result of the binary
     * reduce functor type. The result is written to the first element of the output data.
     * @param binaryReduceFn Reduce binary functor, the functor operation must be transitive and commutative.
     *   The atomic operation atomic::alpakaAtomicInvoke(ReduceFnType, onAcc::concepts::Acc, auto* destination,auto
     * source) must be overloaded. The functor execution order is not specified. The functor should support Simd
     * packages, if not you can enforce the element wise execution by wrapping into
     *   @see ScalarFunc.
     * @param transformFn The function to apply to each element of the input data.
     *   The functor should support Simd packages. If not you can enforce the element wise execution by wrapping into
     * ScalarFunc. If you would like to support stencil executions wrap fn into StencilFunc. StencilFunc is
     * getting all arguments as SimdPtr. If StencilFunc is used you should take care to not read outside of valid
     * memory ranges by using sub-views to your input and output data. Optionally a transformFn can have an accelerator
     * as first argument.
     * @param in The input data to transform, all input data is passed to fn. transformFn must support as many
     * arguments as input data is provided. An optional argument for the accelerator is support as first argument if
     * needed.
     *
     * examples for an identity unary transform functor:
     * @code{.cpp}
     *   struct Foo {
     *      constexpr auto operator()(onAcc::concepts::Acc auto const&, concepts::SimdPtr auto const& a) const {
     *          return a.load();
     *      }
     *   };
     *   struct Bar {
     *      constexpr auto operator()(concepts::SimdPtr auto const& a) const {
     *          return a.load();
     *      }
     *   };
     * @endcode
     *
     * @{
     */
    template<typename T_DataType, typename T_Device, alpaka::concepts::QueuePolicyList T_Policies>
    inline void transformReduce(
        Queue<T_Device, T_Policies> const& queue,
        alpaka::concepts::Executor auto const exec,
        T_DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto out,
        auto&& binaryReduceFn,
        auto&& transformFn,
        alpaka::concepts::IDataSource auto&&... in)
        requires(std::same_as<T_DataType, alpaka::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::transformReduce(
                queue,
                defaultExecutor(queue.getDevice()),
                neutralElement,
                out,
                ALPAKA_FORWARD(binaryReduceFn),
                ALPAKA_FORWARD(transformFn),
                ALPAKA_FORWARD(in)...);
        }
        else
            internal::transformReduce(
                queue,
                exec,
                neutralElement,
                out,
                ALPAKA_FORWARD(binaryReduceFn),
                ALPAKA_FORWARD(transformFn),
                ALPAKA_FORWARD(in)...);
    }

    /**
     * An available default executor will be selected automatically. The default executor is the executor with the most
     * parallelism/performance.
     */
    template<typename T_DataType, typename T_Device, alpaka::concepts::QueuePolicyList T_Policies>
    inline void transformReduce(
        Queue<T_Device, T_Policies> const& queue,
        T_DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto out,
        auto&& binaryReduceFn,
        auto&& transformFn,
        alpaka::concepts::IDataSource auto&&... in)
        requires(std::same_as<T_DataType, alpaka::GetValueType_t<ALPAKA_TYPEOF(out)>>)
    {
        transformReduce(
            queue,
            defaultExecutor(queue.getDevice()),
            neutralElement,
            out,
            ALPAKA_FORWARD(binaryReduceFn),
            ALPAKA_FORWARD(transformFn),
            ALPAKA_FORWARD(in)...);
    }

    /** @} */
} // namespace alpaka::onHost
