/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/functor.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/trait.hpp"

namespace alpaka::onHost::internal
{
    struct SimdTransformKernel
    {
        ALPAKA_FN_ACC void operator()(
            onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::IMdSpan auto&& output,
            auto const& func,
            alpaka::concepts::IDataSource auto&&... inputs) const
        {
            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
            if constexpr(isSpecializationOf_v<ALPAKA_TYPEOF(func), StencilFunc>)
            {
                return simdGrid.concurrent(
                    acc,
                    output.getExtents(),
                    [&](auto const& acc, auto out, auto&&... in)
                    { out = callFunctor(acc, func, ALPAKA_FORWARD(in)...); },
                    ALPAKA_FORWARD(output),
                    ALPAKA_FORWARD(inputs)...);
            }
            else if constexpr(isSpecializationOf_v<ALPAKA_TYPEOF(func), ScalarFunc>)
            {
                simdGrid.concurrent(
                    acc,
                    output.getExtents(),
                    [&](auto const& acc, auto outPtr, auto const&... inPtr) constexpr
                    {
                        outPtr = loadAncExecuteScalarOp(
                            std::make_integer_sequence<uint32_t, ALPAKA_TYPEOF(outPtr)::width()>{},
                            [](alpaka::concepts::CVector auto idx,
                               auto const& acc,
                               auto&& func,
                               auto&&... data) constexpr { return callFunctor(acc, func, data[idx.x()]...); },
                            acc,
                            func,
                            inPtr.load()...);
                    },
                    ALPAKA_FORWARD(output),
                    ALPAKA_FORWARD(inputs)...);
            }
            else
            {
                return simdGrid.concurrent(
                    acc,
                    output.getExtents(),
                    [&](auto const& acc, auto out, auto const&... in) { out = callFunctor(acc, func, in.load()...); },
                    ALPAKA_FORWARD(output),
                    ALPAKA_FORWARD(inputs)...);
            }
        }

        template<uint32_t... T_idx>
        ALPAKA_FN_INLINE static constexpr auto loadAncExecuteScalarOp(
            std::integer_sequence<uint32_t, T_idx...>,
            auto&& op,
            auto const& acc,
            auto&& func,
            auto&&... data)
        {
            return Simd{op(CVec<uint32_t, T_idx>{}, acc, func, data...)...};
        }
    };

    inline void transform(
        auto const& queue,
        alpaka::concepts::Executor auto const exec,
        alpaka::concepts::IMdSpan auto&& out,
        auto&& fn,
        alpaka::concepts::IDataSource auto&&... in)
    {
        auto extentMd = onHost::getExtents(out);
        using DataType = alpaka::GetValueType_t<ALPAKA_TYPEOF(out)>;
        auto frameSpec = getSimdFrameSpec<DataType>(queue.getDevice(), exec, extentMd);

        ALPAKA_LOG_INFO(
            onHost::logger::memory,
            [&]()
            {
                std::stringstream ss;
                ss << "transform{ extents=" << extentMd << ", value_type=" << onHost::demangledName<DataType>() << ", "
                   << frameSpec << ", fn=" << onHost::demangledName(fn) << " }";
                return ss.str();
            });

        queue.enqueue(
            frameSpec,
            KernelBundle{SimdTransformKernel{}, ALPAKA_FORWARD(out), ALPAKA_FORWARD(fn), ALPAKA_FORWARD(in)...});
    }
} // namespace alpaka::onHost::internal
