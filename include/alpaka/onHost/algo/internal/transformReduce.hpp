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
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/trait.hpp"

namespace alpaka::onHost::internal
{
    struct SimdTransformReduceKernel
    {
        uint32_t dynSharedMemBytes = 0u;

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


            // Shared memory for block-wide reduction
            T_DataType* dynS = onAcc::getDynSharedMem<T_DataType>(acc);
            auto pitchMd = alpaka::calculatePitchesFromExtents<T_DataType>(chunkExtents);
            auto tbSum = MdSpan{dynS, chunkExtents, pitchMd};

            auto traverseInFrame = alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::threadsInBlock,
                alpaka::IdxRange{chunkExtents});

            // Initialize shared memory by setting all elements to the neutral element or identity value
            // for the reduction operation.
            for(auto elemIdxInFrame : traverseInFrame)
            {
                tbSum[elemIdxInFrame] = neutralElement;
            }

            auto const chunkDataExtent = numChunks * chunkExtents;
            auto traverseOverFrames = alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::blocksInGrid,
                alpaka::IdxRange{chunkDataExtent.fill(0), chunkDataExtent, chunkExtents});

            for(auto chunkIdx : traverseOverFrames)
            {
                for(alpaka::concepts::Vector auto elemIdxInChunk : traverseInFrame)
                {
                    auto allThreads = alpaka::onAcc::SimdAlgo{
                        alpaka::onAcc::WorkerGroup{chunkIdx + elemIdxInChunk, chunkDataExtent}};

                    // reduce functor with simd package support
                    auto reducedValue
                        = allThreads
                              .transformReduce(acc, extentMd, neutralElement, reduceFunc, transformFunc, inputs...);
                    auto& tbSumRef = tbSum[elemIdxInChunk];
                    tbSumRef = reduceFunc(tbSumRef, reducedValue);
                }
            }

            auto const laneIdInBlock = linearize(acc[alpaka::layer::thread].count(), acc[alpaka::layer::thread].idx());
            auto const blockSize = acc[alpaka::layer::thread].count().product();
            // Synchronize threads before aggregation
            alpaka::onAcc::syncBlockThreads(acc);

            // Aggregate shared memory slots
            for(auto [linearSharedElemIdx] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::IdxRange{blockSize, chunkExtents.product()}))
            {
                dynS[laneIdInBlock] = reduceFunc(dynS[laneIdInBlock], dynS[linearSharedElemIdx]);
            }

            alpaka::onAcc::syncBlockThreads(acc);

            // Perform a parallel reduction within the block
            // This is a tree reduction algorithm
            for(auto offset = blockSize / 2; offset > 0; offset /= 2)
            {
                alpaka::onAcc::syncBlockThreads(acc);
                if(laneIdInBlock < offset)
                {
                    dynS[laneIdInBlock] = reduceFunc(dynS[laneIdInBlock], dynS[laneIdInBlock + offset]);
                }
            }

            // Atomic update of the global result
            if(laneIdInBlock == 0)
            {
                using alpaka::onAcc::atomic::alpakaAtomicInvoke;
                if constexpr(
                    alpaka::concepts::SpecializationOf<ALPAKA_TYPEOF(reduceFunc), ScalarFunc>
                    || alpaka::concepts::SpecializationOf<ALPAKA_TYPEOF(reduceFunc), StencilFunc>)
                {
                    // Handle wrapped reduce functors e.g. ScalarFunc or StencilFunc
                    using ReduceFunctor = typename ALPAKA_TYPEOF(reduceFunc)::Functor;
                    alpakaAtomicInvoke(
                        static_cast<ReduceFunctor const&>(reduceFunc),
                        acc,
                        output.data(),
                        dynS[laneIdInBlock]);
                }
                else
                    alpakaAtomicInvoke(reduceFunc, acc, output.data(), dynS[laneIdInBlock]);
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

        auto kernelFn = SimdTransformReduceKernel{
            static_cast<uint32_t>(frameSpec.getFrameExtents().product() * sizeof(T_DataType))};

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
