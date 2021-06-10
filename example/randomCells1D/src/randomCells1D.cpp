/* Copyright 2021 Jiri Vyskocil
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

unsigned constexpr NUM_CALCULATIONS = 256;
//unsigned constexpr NUM_X = 100000;    // works on CUDA
unsigned constexpr NUM_X = 10000;   // fails on CUDA

struct InitRandomKernel
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TExtent const extent, TRandEngine* const states) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const linearIdx = alpaka::mapIdx<1u>(idx, extent)[0];
        TRandEngine engine(42, static_cast<std::uint32_t>(linearIdx));
        states[linearIdx] = engine;
    }
};

struct RunTimestepKernelSingle
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        alpaka::rand::Philox4x32x10<TAcc>* const states,
        float* const cellsA) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const linearIdx = alpaka::mapIdx<1u>(idx, extent)[0];

        // Setup generator and distribution.
        alpaka::rand::Philox4x32x10<TAcc> engine(states[linearIdx]);
        alpaka::rand::UniformReal<float> dist;

        float sumA = 0;
        for(unsigned numCalculations = 0; numCalculations < NUM_CALCULATIONS; ++numCalculations)
        {
            sumA += dist(engine);
        }
        cellsA[linearIdx] = sumA / NUM_CALCULATIONS;
        states[linearIdx] = engine;
    }
};

struct RunTimestepKernelVector
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        alpaka::rand::Philox4x32x10Vector<TAcc>* const states,
        float* const cellsB) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const linearIdx = alpaka::mapIdx<1u>(idx, extent)[0];

        // Setup generator and distribution.
        alpaka::rand::Philox4x32x10Vector<TAcc> engine(states[linearIdx]);
        using RandResultVector = decltype(engine());
        unsigned constexpr resultVectorSize = std::tuple_size_v<RandResultVector>;
        alpaka::rand::UniformReal<std::array<float, resultVectorSize>> dist;

        float sumB = 0;
        static_assert(NUM_CALCULATIONS % resultVectorSize == 0);
        for(unsigned numCalculations = 0; numCalculations < NUM_CALCULATIONS / resultVectorSize; ++numCalculations)
        {
            auto result = dist(engine);
            for(int i = 0; i < static_cast<int>(resultVectorSize); ++i)
            {
                sumB += result[i];
            }
        }
        cellsB[linearIdx] = sumB / NUM_CALCULATIONS;
        states[linearIdx] = engine;
    }
};

auto main() -> int
{
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto const devHost = alpaka::getDevByIdx<Host>(0u);
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{devAcc};

    using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;
    using BufHostRand = alpaka::Buf<Host, alpaka::rand::Philox4x32x10<Acc>, Dim, Idx>;
    using BufAccRand = alpaka::Buf<Acc, alpaka::rand::Philox4x32x10<Acc>, Dim, Idx>;
    using BufHostRandVec = alpaka::Buf<Host, alpaka::rand::Philox4x32x10Vector<Acc>, Dim, Idx>;
    using BufAccRandVec = alpaka::Buf<Acc, alpaka::rand::Philox4x32x10Vector<Acc>, Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    constexpr Idx numX = NUM_X;

    const Vec extent(numX);

    constexpr Idx perThreadX = 1;

    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
        devAcc,
        extent,
        Vec(perThreadX),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    // Setup buffer.
    BufHost bufHostA{alpaka::allocBuf<float, Idx>(devHost, extent)};
    float* const ptrBufHostA{alpaka::getPtrNative(bufHostA)};
    BufAcc bufAccA{alpaka::allocBuf<float, Idx>(devAcc, extent)};
    float* const ptrBufAccA{alpaka::getPtrNative(bufAccA)};

    BufHost bufHostB{alpaka::allocBuf<float, Idx>(devHost, extent)};
    float* const ptrBufHostB{alpaka::getPtrNative(bufHostB)};
    BufAcc bufAccB{alpaka::allocBuf<float, Idx>(devAcc, extent)};
    float* const ptrBufAccB{alpaka::getPtrNative(bufAccB)};

    BufHostRand bufHostRandA{alpaka::allocBuf<alpaka::rand::Philox4x32x10<Acc>, Idx>(devHost, extent)};
    BufAccRand bufAccRandA{alpaka::allocBuf<alpaka::rand::Philox4x32x10<Acc>, Idx>(devAcc, extent)};
    alpaka::rand::Philox4x32x10<Acc>* const ptrBufAccRandA{alpaka::getPtrNative(bufAccRandA)};

    BufHostRandVec bufHostRandB{alpaka::allocBuf<alpaka::rand::Philox4x32x10Vector<Acc>, Idx>(devHost, extent)};
    BufAccRandVec bufAccRandB{alpaka::allocBuf<alpaka::rand::Philox4x32x10Vector<Acc>, Idx>(devAcc, extent)};
    alpaka::rand::Philox4x32x10Vector<Acc>* const ptrBufAccRandB{alpaka::getPtrNative(bufAccRandB)};

    InitRandomKernel initRandomKernel;
    alpaka::exec<Acc>(queue, workdiv, initRandomKernel, extent, ptrBufAccRandA);
    alpaka::wait(queue);

    alpaka::exec<Acc>(queue, workdiv, initRandomKernel, extent, ptrBufAccRandB);
    alpaka::wait(queue);

    for(Idx x = 0; x < numX; ++x)
    {
        ptrBufHostA[x] = 0;
        ptrBufHostB[x] = 0;
    }

    alpaka::memcpy(queue, bufAccA, bufHostA, extent);
    RunTimestepKernelSingle runTimestepKernelSingle;
    alpaka::exec<Acc>(queue, workdiv, runTimestepKernelSingle, extent, ptrBufAccRandA, ptrBufAccA);
    alpaka::memcpy(queue, bufHostA, bufAccA, extent);


    alpaka::memcpy(queue, bufAccB, bufHostB, extent);
    RunTimestepKernelVector runTimestepKernelVector;
    alpaka::exec<Acc>(queue, workdiv, runTimestepKernelVector, extent, ptrBufAccRandB, ptrBufAccB);
    alpaka::memcpy(queue, bufHostB, bufAccB, extent);
    alpaka::wait(queue);

    float avgA = std::reduce(ptrBufHostA, ptrBufHostA + numX);
    float avgB = std::reduce(ptrBufHostB, ptrBufHostB + numX);

    avgA /= numX;
    avgB /= numX;

    std::cout << "Number of cells: " << numX << "\n";
    std::cout << "Number of calculations: " << NUM_CALCULATIONS << "\n";
    std::cout << "Mean value A: " << avgA << "\n";
    std::cout << "Mean value B: " << avgB << "\n";

    return 0;
}
