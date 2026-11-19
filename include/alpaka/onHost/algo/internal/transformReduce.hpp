/* Copyright 2025 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/functor.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/mem/concepts/IDataSource.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/atomic.hpp"
#include "alpaka/onAcc/warp.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/trait.hpp"

namespace alpaka::onHost::internal
{
    struct SimdTransformReduceKernel
    {
        uint32_t dynSharedMemBytes = 0u;

        // Atomic update of the global result, wrapped reduce functors e.g. ScalarFunc or StencilFunc are unwrapped.
        static ALPAKA_FN_ACC void atomicUpdate(
            auto const& acc,
            auto const& reduceFunc,
            auto* result,
            auto const& value)
        {
            using alpaka::onAcc::atomic::alpakaAtomicInvoke;
            if constexpr(
                alpaka::concepts::SpecializationOf<ALPAKA_TYPEOF(reduceFunc), ScalarFunc>
                || alpaka::concepts::SpecializationOf<ALPAKA_TYPEOF(reduceFunc), StencilFunc>)
            {
                using ReduceFunctor = typename ALPAKA_TYPEOF(reduceFunc)::Functor;
                alpakaAtomicInvoke(static_cast<ReduceFunctor const&>(reduceFunc), acc, result, value);
            }
            else
                alpakaAtomicInvoke(reduceFunc, acc, result, value);
        }

        template<typename T_DataType>
        ALPAKA_FN_ACC void operator()(
            onAcc::concepts::Acc auto const& acc,
            alpaka::concepts::Vector auto const& numChunks,
            alpaka::concepts::Vector auto const& chunkExtents,
            alpaka::concepts::Vector auto const& extentMd,
            T_DataType const& neutralElement,
            alpaka::concepts::IMdSpan auto output,
            auto const& reduceFunc,
            auto const& transformFunc,
            alpaka::concepts::IDataSource auto&&... inputs) const
        {
            static_assert(
                std::is_same_v<ALPAKA_TYPEOF(neutralElement), alpaka::GetValueType_t<ALPAKA_TYPEOF(output)>>,
                "The neutral element type must match the data output type.");


            /* Each thread reduces its elements in a register. The threads of a warp are reduced with warp shuffles,
             * only the warp results are exchanged via shared memory.
             */
            auto const chunkDataExtent = numChunks * chunkExtents;
            auto traverseOverFrames = alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::blocksInGrid,
                alpaka::IdxRange{chunkDataExtent.fill(0), chunkDataExtent, chunkExtents});
            auto traverseInFrame = alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::threadsInBlock,
                alpaka::IdxRange{chunkExtents});

            T_DataType threadResult = neutralElement;
            for(auto chunkIdx : traverseOverFrames)
            {
                for(alpaka::concepts::Vector auto elemIdxInChunk : traverseInFrame)
                {
                    auto allThreads = alpaka::onAcc::SimdAlgo{
                        alpaka::onAcc::WorkerGroup{chunkIdx + elemIdxInChunk, chunkDataExtent}};

                    // reduce functor with simd package support
                    threadResult = reduceFunc(
                        threadResult,
                        allThreads
                            .transformReduce(acc, extentMd, neutralElement, reduceFunc, transformFunc, inputs...));
                }
            }

            // The block size is a power of two, as required by warp::reduce for blocks smaller than a warp.
            T_DataType const warpResult = onAcc::warp::reduce(acc, threadResult, reduceFunc);

            T_DataType* warpResults = onAcc::getDynSharedMem<T_DataType>(acc);
            if(onAcc::warp::getLaneIdx(acc) == 0u)
                warpResults[onAcc::warp::getWarpIdx(acc)] = warpResult;
            alpaka::onAcc::syncBlockThreads(acc);

            // The first warp reduces the warp results and updates the global result.
            if(onAcc::warp::getWarpIdx(acc) == 0u)
            {
                T_DataType blockResult = neutralElement;
                for(auto [i] :
                    onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInWarp, onAcc::range::linearWarpsInBlock))
                    blockResult = reduceFunc(blockResult, warpResults[i]);
                blockResult = onAcc::warp::reduce(acc, blockResult, reduceFunc);

                if(onAcc::warp::getLaneIdx(acc) == 0u)
                    atomicUpdate(acc, reduceFunc, output.data(), blockResult);
            }
        }
    };

    template<typename T_DataType>
    inline void transformReduce(
        auto const& queue,
        alpaka::concepts::Executor auto const exec,
        T_DataType const& neutralElement,
        alpaka::concepts::IMdSpan auto&& out,
        auto&& reduceFn,
        auto&& transformFn,
        auto&& in0,
        alpaka::concepts::IDataSource auto&&... in)
    {
        auto extentMd = onHost::getExtents(in0);
        using IndexType = alpaka::GetValueType_t<ALPAKA_TYPEOF(extentMd)>;
        auto frameSpec = getSimdFrameSpec<T_DataType>(queue.getDevice(), exec, extentMd);

        /* Adjust the launch parameters to not oversubscribe a device too much.
         *
         * @todo: This heuristic should be adjusted based on benchmarking different cases.
         */
        {
            IndexType multiprocessorScaling = 1u;
            if constexpr(!(ALPAKA_TYPEOF(queue.getDevice().getDeviceKind()){} == deviceKind::cpu))
            {
                // For non-CPU devices, we scale the number of frames based on an arbitrary number derived from
                // testing with the dot kernel of the bablestream benchmark.
                multiprocessorScaling = 32u;
            }

            auto const numMultiProcessors = queue.getDevice().getDeviceProperties().multiProcessorCount;
            auto adjsutedNumFrames = alpaka::api::util::adjustToLimit(
                frameSpec.getNumFrames(),
                static_cast<IndexType>(numMultiProcessors * multiprocessorScaling));
            frameSpec = FrameSpec{adjsutedNumFrames, frameSpec.getFrameExtents(), exec};
        }

        /* Derive the chunk size and number of chunks from the SIMD optimized frame specification.
         * The chunking parameters influences the numerical precision because it provides the possibility to control
         * the length of the accumulation chain of a single thread.
         */
        auto numChunks = frameSpec.getNumFrames();
        auto chunkExtents = frameSpec.getFrameExtents();

        // shared memory holds one result per warp
        auto const warpSize = static_cast<IndexType>(queue.getDevice().getDeviceProperties().warpSize);
        auto kernelFn = SimdTransformReduceKernel{
            static_cast<uint32_t>(divCeil(chunkExtents.product(), warpSize) * sizeof(T_DataType))};

        ALPAKA_LOG_INFO(
            onHost::logger::memory,
            [&]()
            {
                std::stringstream ss;
                ss << "transformReduce{ extents=" << extentMd << ", value_type=" << onHost::demangledName<T_DataType>()
                   << ", " << frameSpec << ", reduceFn=" << onHost::demangledName(reduceFn)
                   << ", transformFn=" << onHost::demangledName(transformFn) << " }";
                return ss.str();
            });

        onHost::fill(queue, ALPAKA_FORWARD(out), neutralElement, out.getExtents().fill(1));
        queue.enqueue(
            frameSpec,
            KernelBundle{
                kernelFn,
                numChunks,
                chunkExtents,
                extentMd,
                neutralElement,
                ALPAKA_FORWARD(out),
                ALPAKA_FORWARD(reduceFn),
                ALPAKA_FORWARD(transformFn),
                ALPAKA_FORWARD(in0),
                ALPAKA_FORWARD(in)...});
    }
} // namespace alpaka::onHost::internal
