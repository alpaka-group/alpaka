/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/CVec.hpp"
#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/warp.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/trait.hpp"

#include <array> // std::array
#include <cstddef> // std::size_t
#include <tuple> // std::tuple
#include <type_traits> // is_same_v
#include <typeinfo>

namespace alpaka::onHost::internal
{
    enum ScanType
    {
        exclusiveScan,
        inclusiveScan
    };

    constexpr std::size_t chunkSize = 2048u;

    template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Idx, typename T_Data>
    consteval T_Idx maximumMiniBlockSize()
    {
        if constexpr(T_DeviceKind{} == deviceKind::nvidiaGpu)
            return static_cast<T_Idx>(8);
        else if constexpr(T_DeviceKind{} == deviceKind::amdGpu)
            return static_cast<T_Idx>(8);
        else if constexpr(T_DeviceKind{} == deviceKind::intelGpu)
            return static_cast<T_Idx>(8);
        else
            return static_cast<T_Idx>(32768) / sizeof(T_Data);
    }

    /* This function introduces padding to the shared memory accesses to reduce bank conflicts between threads. The
     * template parameter is the device kind, which dictates how many memory banks are assumed. For CPU or
     * unknown/unimplemented device kinds, infinite memory banks are assumed, i.e., no padding is used.
     */
    template<typename T_Acc, typename T_Idx>
    constexpr T_Idx conflictFreeAccess(T_Idx const& n)
    {
        constexpr auto warpSize = static_cast<T_Idx>(onAcc::warp::getSize<T_Acc>());
        return n + n / warpSize;
    }

    /* Do a muting exclusive scan on the given miniblock, and return the total sum.
     */
    template<typename T_Idx, typename T_Data>
    ALPAKA_FN_ACC T_Data scanMiniBlock(T_Data* block, alpaka::concepts::CVector<T_Idx> auto const& extent)
    {
        // -- UP-SWEEP / REDUCE --
        for(T_Idx d = extent.x() / T_Idx{2}, offset = T_Idx{1}; d > 0; d >>= 1, offset <<= 1)
        {
            for(auto frameElem = T_Idx{0}; frameElem < T_Idx{2} * d; frameElem += T_Idx{2})
            {
                T_Idx left = offset * (frameElem + T_Idx{1}) - T_Idx{1};
                T_Idx right = offset * (frameElem + T_Idx{2}) - T_Idx{1};
                block[right] += block[left];
            }
        }

        // save total sum
        T_Data blockSum = block[extent.x() - T_Idx{1}];

        // set 0
        block[extent.x() - T_Idx{1}] = T_Data{0};

        // -- DOWN-SWEEP --
        for(T_Idx d = 1, offset = extent.x() / T_Idx{2}; d < extent.x(); d <<= 1, offset >>= 1)
        {
            for(auto frameElem = T_Idx{0}; frameElem < T_Idx{2} * d; frameElem += T_Idx{2})
            {
                T_Idx left = offset * (frameElem + T_Idx{1}) - T_Idx{1};
                T_Idx right = offset * (frameElem + T_Idx{2}) - T_Idx{1};
                auto t = block[left];
                block[left] = block[right];
                block[right] += t;
            }
        }
        return blockSum;
    }

    /* Do an add increment on the given miniblock, adding the given blockSum to each element.
     */
    template<typename T_Idx, typename T_Data>
    ALPAKA_FN_ACC void addIncrements(
        T_Data* block,
        T_Data const& blockSum,
        alpaka::concepts::CVector<T_Idx> auto const& extent)
    {
        for(auto i = T_Idx{0}; i < extent.x(); ++i)
        {
            block[i] += blockSum;
        }
    }

    /* This kernel calculates an exclusive scan for each block individually. The algorithm is based on Blelloch, with
     * the improvement from Lichterman, written up in the CUDA blog (see 39.2.5):
     * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
     */
    template<ScanType T_scanType, typename T_Idx, typename T_Data>
    class ScanBlocksKernel
    {
    public:
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::Vector auto const numChunks,
            alpaka::concepts::CVector auto const largeChunkExtents,
            alpaka::concepts::IDataSource auto const& inputVec,
            alpaka::concepts::IMdSpan auto outputVec,
            auto... blockSums) const
        {
            using DeviceType = ALPAKA_TYPEOF(acc.getDeviceKind());
            using AccType = ALPAKA_TYPEOF(acc);

            alpaka::concepts::CVector auto numThreadsPerBlock = acc[layer::thread].count();
            constexpr std::integral auto elsPerThread = largeChunkExtents.x() / numThreadsPerBlock.x();
            alpaka::concepts::CVector auto chunkExtent = CVec<T_Idx, elsPerThread * numThreadsPerBlock.x()>{};
            alpaka::concepts::Vector auto numElements = inputVec.getExtents();

            constexpr std::integral auto miniBlockSize
                = std::min(maximumMiniBlockSize<DeviceType, T_Idx, T_Data>(), elsPerThread);
            constexpr std::integral auto miniBlocksPerThread = elsPerThread / miniBlockSize;
            constexpr std::integral auto miniBlocksPerChunk = chunkExtent.x() / miniBlockSize;

            constexpr auto lLocalArrayLength = miniBlocksPerThread * miniBlockSize;
            using LocalArray = T_Data[lLocalArrayLength];

            auto const validElementsInLastFrame = (numElements - T_Idx{1}) % chunkExtent + T_Idx{1};

            /* This kernel is called with 1-dimensional frame extents.
             *
             * All thread blocks will be used to iterate over the frames. Each thread block will handle one or more
             * frames.
             */
            for(auto chunkIdx :
                onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec<T_Idx, 1u>{0}, numChunks}))
            {
                bool const lastFrameFull = validElementsInLastFrame == chunkExtent;
                bool const isLastFrame = chunkIdx == numChunks - T_Idx{1};

                // allocate "per-thread" register memory to store all mini blocks of a thread persistently
                LocalArray regMem;

                constexpr auto conflictFreeAdr = conflictFreeAccess<AccType>(miniBlocksPerChunk - T_Idx{1}) + T_Idx{1};
                auto tmp = onAcc::declareSharedMdArray<T_Data, uniqueId()>(acc, CVec<T_Idx, conflictFreeAdr>{});
                auto const frameOffset = chunkExtent * chunkIdx;

                for(auto frameElem : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<T_Idx, 0u>{}, chunkExtent, CVec<T_Idx, elsPerThread>{}}))
                {
                    // -- COPY TO SHARED MEM --
                    if((!lastFrameFull && isLastFrame) || elsPerThread % T_Idx{4} != T_Idx{0})
                    {
                        // load into miniblocks buffer, from frameElem to frameElem + elsPerThread
                        for(auto i = T_Idx{0}; i < elsPerThread; ++i)
                        {
                            if(frameOffset + frameElem + i < numElements)
                                regMem[i] = inputVec[frameOffset + frameElem + i];
                            else
                                regMem[i] = 0;
                        }
                    }
                    else
                    {
                        MdSpanArray<LocalArray, T_Idx, alpaka::Alignment<16>> regMemMd{regMem};

                        for(auto i = T_Idx{0}; i < elsPerThread; i += T_Idx{4})
                        {
                            auto inputVecView = SimdPtr{
                                inputVec,
                                Vec{frameOffset + frameElem + i},
                                Alignment<16>{},
                                CVec<T_Idx, 4>{}};
                            auto regView = SimdPtr{regMemMd, Vec{i}, Alignment<16>{}, CVec<T_Idx, 4>{}};

                            regView = inputVecView.load();
                        }
                    }

                    // -- HANDLE MINI BLOCKS OF THIS THREAD --
                    for(auto miniBlockOffset = T_Idx{0}; miniBlockOffset < elsPerThread;
                        miniBlockOffset += miniBlockSize)
                    {
                        // scan miniblock
                        auto miniBlockSum
                            = scanMiniBlock<T_Idx, T_Data>(regMem + miniBlockOffset, CVec<T_Idx, miniBlockSize>{});

                        // write miniblock sum into shared memory
                        tmp[conflictFreeAccess<AccType>((frameElem + miniBlockOffset) / miniBlockSize)] = miniBlockSum;
                    }
                }

                // -- UP-SWEEP / REDUCE --
                for(T_Idx d = miniBlocksPerChunk / T_Idx{2}, offset = T_Idx{1}; d > 0; d >>= 1, offset <<= 1)
                {
                    onAcc::syncBlockThreads(acc);
                    for(auto frameElem : onAcc::makeIdxMap(
                            acc,
                            onAcc::worker::threadsInBlock,
                            IdxRange{CVec<T_Idx, 0>{}, Vec<T_Idx, 1>{T_Idx{2} * d}, T_Idx{2}}))
                    {
                        T_Idx left = offset * (frameElem + T_Idx{1}).x() - T_Idx{1};
                        T_Idx right = offset * (frameElem + T_Idx{2}).x() - T_Idx{1};
                        left = conflictFreeAccess<AccType>(left);
                        right = conflictFreeAccess<AccType>(right);
                        tmp[right] += tmp[left];
                    }
                }
                onAcc::syncBlockThreads(acc);

                for([[maybe_unused]] auto frameElem :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1}))
                {
                    // -- SAVE BLOCK SUMS --
                    if constexpr(sizeof...(blockSums))
                    {
                        auto blockSums0 = std::get<0>(std::make_tuple(blockSums...));
                        blockSums0[chunkIdx] = tmp[conflictFreeAccess<AccType>(miniBlocksPerChunk - T_Idx{1})];
                    }

                    // -- SET 0 --
                    tmp[conflictFreeAccess<AccType>(miniBlocksPerChunk - T_Idx{1})] = 0;
                }

                // -- DOWN-SWEEP --
                for(T_Idx d = 1, offset = miniBlocksPerChunk / T_Idx{2}; d < miniBlocksPerChunk; d <<= 1, offset >>= 1)
                {
                    onAcc::syncBlockThreads(acc);
                    for(auto frameElem : onAcc::makeIdxMap(
                            acc,
                            onAcc::worker::threadsInBlock,
                            IdxRange{CVec<T_Idx, 0>{}, Vec<T_Idx, 1>{T_Idx{2} * d}, T_Idx{2}}))
                    {
                        T_Idx left = offset * (frameElem.x() + T_Idx{1}) - T_Idx{1};
                        T_Idx right = offset * (frameElem.x() + T_Idx{2}) - T_Idx{1};
                        left = conflictFreeAccess<AccType>(left);
                        right = conflictFreeAccess<AccType>(right);
                        auto t = tmp[left];
                        tmp[left] = tmp[right];
                        tmp[right] += t;
                    }
                }
                onAcc::syncBlockThreads(acc);

                // -- WRITE BACK --
                for(auto frameElem : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::threadsInBlock,
                        IdxRange{CVec<T_Idx, 0u>{}, chunkExtent, CVec<T_Idx, elsPerThread>{}}))
                {
                    // -- HANDLE MINI BLOCKS OF THIS THREAD --
                    for(auto miniBlockOffset = T_Idx{0}; miniBlockOffset < elsPerThread;
                        miniBlockOffset += miniBlockSize)
                    {
                        // load block sum from shared memory
                        T_Data blockSum{0};
                        if(frameOffset + frameElem + miniBlockOffset < numElements)
                        {
                            blockSum
                                = tmp[conflictFreeAccess<AccType>((frameElem.x() + miniBlockOffset) / miniBlockSize)];
                        }

                        // add block sum to mini block
                        addIncrements<T_Idx>(regMem + miniBlockOffset, blockSum, CVec<T_Idx, miniBlockSize>{});
                    }

                    if((!lastFrameFull && isLastFrame) || elsPerThread % T_Idx{4} != T_Idx{0})
                    {
                        // write back to global mem, from frameElem to frameElem + elsPerThread
                        for(auto i = T_Idx{0}; i < elsPerThread; ++i)
                        {
                            if(frameOffset + frameElem + i < numElements)
                            {
                                if constexpr(T_scanType == exclusiveScan)
                                    outputVec[frameOffset + frameElem + i] = regMem[i];
                                else if constexpr(T_scanType == inclusiveScan)
                                    outputVec[frameOffset + frameElem + i]
                                        = inputVec[frameOffset + frameElem + i] + regMem[i];
                            }
                        }
                    }
                    else
                    {
                        MdSpanArray<LocalArray, T_Idx, alpaka::Alignment<16>> regMemMd{regMem};

                        for(auto i = T_Idx{0}; i < elsPerThread; i += T_Idx{4})
                        {
                            auto outputVecView = SimdPtr{
                                outputVec,
                                Vec{frameOffset + frameElem + i},
                                Alignment<16>{},
                                CVec<T_Idx, 4>{}};
                            auto regView = SimdPtr{regMemMd, Vec{i}, Alignment<16>{}, CVec<T_Idx, 4>{}};
                            if constexpr(T_scanType == exclusiveScan)
                                outputVecView = regView.load();
                            else if constexpr(T_scanType == inclusiveScan)
                            {
                                auto inputVecView = SimdPtr{
                                    inputVec,
                                    Vec{frameOffset + frameElem + i},
                                    Alignment<16>{},
                                    CVec<T_Idx, 4>{}};
                                outputVecView = inputVecView.load() + regView.load();
                            }
                        }
                    }
                }
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    /* Add prefix sum from previous blocks (blockSums) to all elements in each block.
     */
    template<typename T_Idx>
    class ScanAddIncrementsKernel
    {
    public:
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::CVector auto const largeChunkExtents,
            alpaka::concepts::IMdSpan auto const& blockSums,
            alpaka::concepts::IMdSpan auto outputVec) const
        {
            alpaka::concepts::Vector auto numElements = outputVec.getExtents();
            alpaka::concepts::CVector auto numThreadsPerBlock = acc[layer::thread].count();
            constexpr auto elsPerThread = largeChunkExtents.x() / numThreadsPerBlock.x();
            alpaka::concepts::CVector auto chunkExtent = CVec<T_Idx, elsPerThread * numThreadsPerBlock.x()>{};

            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
            simdGrid.concurrent(
                acc,
                numElements,
                [&](auto const&, auto&& simdOut) constexpr
                { simdOut = simdOut.load() + blockSums[simdOut.getIdx() / chunkExtent]; },
                outputVec);
        }
    };

    template<typename T_Data>
    auto scanBufferSize(std::integral auto const& extent)
    {
        using Idx = ALPAKA_TYPEOF(extent);
        auto elements = divCeil(extent, Idx{chunkSize});

        auto bufSize = Idx{0};
        while(elements > Idx{1})
        {
            bufSize += elements;
            elements = divCeil(elements, Idx{chunkSize});
        }

        return bufSize * Idx{sizeof(T_Data)};
    }

    template<typename T_Data>
    auto scanBufferSize(alpaka::concepts::Vector auto const& extents)
    {
        static_assert(ALPAKA_TYPEOF(extents)::dim() == 1, "scan is only usable for one dimensional buffers");
        return Vec{scanBufferSize<T_Data>(extents.x())};
    }

    template<ScanType T_scanType>
    void scan(
        auto& queue,
        alpaka::onHost::concepts::Device auto& devAcc,
        alpaka::concepts::Executor auto& exec,
        alpaka::concepts::IMdSpan auto&& buffer,
        alpaka::concepts::IMdSpan auto&& outputVec,
        alpaka::concepts::IDataSource auto& inputVec)
    {
        using Data = typename ALPAKA_TYPEOF(inputVec)::value_type;
        using Idx = typename ALPAKA_TYPEOF(inputVec)::index_type;

        static_assert(
            std::is_same_v<Data, typename ALPAKA_TYPEOF(outputVec)::value_type>,
            "output vector must have the same data type as input vector");

        // Instantiate the kernel function object with the given scan type
        ScanBlocksKernel<T_scanType, Idx, Data> scanBlocks;

        // Define chunkExtent
        constexpr auto chunkExtent = CVec<Idx, chunkSize>{};
        alpaka::Vec numChunks = divCeil(inputVec.getExtents(), chunkExtent);
        auto const frameSpec = onHost::FrameSpec{numChunks, CVec<Idx, 256u>{}};

        ALPAKA_LOG_INFO(
            onHost::logger::memory,
            [&]()
            {
                std::stringstream ss;
                ss << "scan: {";
                if(T_scanType == inclusiveScan)
                    ss << ", scanType= INCLUSIVE_SCAN";
                else if(T_scanType == exclusiveScan)
                    ss << ", scanType= EXCLUSIVE_SCAN";
                ss << ", numFrames= " << numChunks;
                ss << ", chunkExtent= " << chunkExtent;
                ss << ", value_type=" << onHost::demangledName<Data>();
                ss << "}";
                return ss.str();
            });

        if(frameSpec.getNumFrames() > Idx{1})
        {
            // problem does not fit in 1 frame, recurse
            ScanAddIncrementsKernel<Idx> addIncrements;

            auto bufSizeBytes = frameSpec.getNumFrames() * Idx{sizeof(Data)};
            assert(buffer.getExtents() * Idx{sizeof(typename ALPAKA_TYPEOF(buffer)::value_type)} >= bufSizeBytes);

            // get the view to the necessary elements in the buffer for increments
            auto subBuf = buffer.getSubView(bufSizeBytes);
            auto increments
                = MdSpan{reinterpret_cast<Data*>(subBuf.data()), frameSpec.getNumFrames(), Vec<Idx, 1>{sizeof(Data)}};

            // the unused elements in the buffer are used for recursion to the next scan call
            auto bufferNext = buffer.getSubView(bufSizeBytes, buffer.getExtents() - bufSizeBytes);

            // enqueue the kernel execution tasks
            queue.enqueue(
                frameSpec,
                KernelBundle{scanBlocks, numChunks, chunkExtent, inputVec, ALPAKA_FORWARD(outputVec), increments});

            // always recurse into exclusive scan
            scan<exclusiveScan>(queue, devAcc, exec, bufferNext, increments, increments);
            queue.enqueue(frameSpec, KernelBundle{addIncrements, chunkExtent, increments, ALPAKA_FORWARD(outputVec)});
        }
        else
        {
            // problem fits within 1 frame
            queue.enqueue(
                frameSpec,
                KernelBundle{scanBlocks, numChunks, chunkExtent, inputVec, ALPAKA_FORWARD(outputVec)});
        }
    }

    template<ScanType T_scanType>
    void scan(
        auto& queue,
        alpaka::onHost::concepts::Device auto& devAcc,
        alpaka::concepts::Executor auto& exec,
        alpaka::concepts::IMdSpan auto&& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        using Data = ALPAKA_TYPEOF(inputVec)::value_type;

        /* We do not use allocDeferred here since we measured up to a factor 40 higher latency compared to alloc for
         * CUDA 12.8 on an A30 for the first call. The reason is the cuda per stream caching pool setup time.
         */
        auto buf = onHost::alloc<char>(devAcc, scanBufferSize<Data>(inputVec.getExtents()));

        scan<T_scanType>(queue, devAcc, exec, buf, ALPAKA_FORWARD(outputVec), inputVec);

        buf.keepAlive(queue);
    }

} // namespace alpaka::onHost::internal
