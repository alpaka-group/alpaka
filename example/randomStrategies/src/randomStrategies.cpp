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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// This example generates NUM_ROLLS of random events for each of NUM_POINTS points.
unsigned constexpr NUM_POINTS = 2000; ///< Number of "points". Each will be  processed by a single thread.
unsigned constexpr NUM_ROLLS = 2000; ///< Amount of random number "dice rolls" performed for each "point".

/// PRNG result space division strategy
enum struct Strategy
{
    seed, ///< threads start from different seeds
    subsequence, ///< threads use different subsequences
    offset ///< threads skip a number of elements in the sequence
};

/// Set initial values for the PRNG states. These will be later advanced by the FillKernel
template<Strategy TStrategy>
struct InitRandomKernel
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the PRNG states buffer
        TRandEngine* const states, ///< PRNG states buffer
        unsigned const skipLength = 0 ///< number of PRNG elements to skip (offset strategy only)
    ) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; ///< index of the current thread

        // Initial engine parameters are to zero. The non-zero parameter corresponding to the chosen strategy will
        // be overridden later.
        std::uint32_t seed{0};
        std::uint32_t subsequence{0};
        std::uint32_t offset{0};

        // Set the initial parameter based on the chosen strategy
        if constexpr(TStrategy == Strategy::seed)
            seed = idx; // each thread starts from a different seed
        else if constexpr(TStrategy == Strategy::subsequence)
            subsequence = idx; // each thread uses a different subsequence
        else if constexpr(TStrategy == Strategy::offset)
            offset = idx * skipLength; // each thread skips a number of elements in the sequence

        TRandEngine engine(seed, subsequence, offset); // Initialize the engine
        states[idx] = engine; // Save the initial state
    }
};

/// Fill the result buffer with random "dice rolls"
struct FillKernel
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the results buffer
        alpaka::rand::Philox4x32x10<TAcc>* const states, ///< PRNG states buffer
        float* const cells ///< results buffer
    ) const -> void
    {
        /// Index of the current thread. Each thread performs multiple "dice rolls".
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        /// Number of "dice rolls" to be performed by each thread.
        auto const length = extent[0] / alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        alpaka::rand::Philox4x32x10<TAcc> engine(states[idx]); // Setup the PRNG using the saved state for this thread.
        alpaka::rand::UniformReal<float> dist; // Setup the random number distribution
        for(unsigned i = 0; i < length; ++i)
        {
            cells[length * idx + i] = dist(engine); // Roll the dice!
        }
        states[idx] = engine; // Save the final PRNG state
    }
};

/// Contains the parameters to set up the default accelerator, queue, and buffers
struct Box
{
    // accelerator, queue, and work division typedefs
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    QueueAcc queue; ///< default accelerator queue

    // buffers holding the PRNG states
    using BufHostRand = alpaka::Buf<Host, alpaka::rand::Philox4x32x10<Acc>, Dim, Idx>;
    using BufAccRand = alpaka::Buf<Acc, alpaka::rand::Philox4x32x10<Acc>, Dim, Idx>;

    Vec const extentRand; ///< size of the buffer of PRNG states
    WorkDiv workdivRand; ///< work division for PRNG buffer initialization
    BufHostRand bufHostRand; ///< host side PRNG states buffer (can be used to check the state of the states)
    BufAccRand bufAccRand; ///< device side PRNG states buffer

    // buffers holding the "simulation" results
    using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;

    Vec const extentResult; ///< size of the results buffer
    WorkDiv workdivResult; ///< work division of the result calculation
    BufHost bufHostResult; ///< host side results buffer
    BufAcc bufAccResult; ///< device side results buffer

    Box()
        : queue{alpaka::getDevByIdx<Acc>(0u)}
        , extentRand{static_cast<Idx>(NUM_POINTS)} // One PRNG state per "point".
        , workdivRand{alpaka::getValidWorkDiv<Acc>(
              alpaka::getDevByIdx<Acc>(0u),
              extentRand,
              Vec(1ul),
              false,
              alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)}
        , bufHostRand{alpaka::allocBuf<alpaka::rand::Philox4x32x10<Acc>, Idx>(
              alpaka::getDevByIdx<Host>(0u),
              extentRand)}
        , bufAccRand{alpaka::allocBuf<alpaka::rand::Philox4x32x10<Acc>, Idx>(alpaka::getDevByIdx<Acc>(0u), extentRand)}
        , extentResult{static_cast<Idx>((NUM_POINTS * NUM_ROLLS))} // Store all "rolls" for each "point"
        , workdivResult{alpaka::getValidWorkDiv<Acc>(
              alpaka::getDevByIdx<Acc>(0u),
              extentResult,
              Vec(static_cast<Idx>(NUM_ROLLS)), // One thread per "point"; each performs NUM_ROLLS "rolls"
              false,
              alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)}
        , bufHostResult{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), extentResult)}
        , bufAccResult{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), extentResult)}
    {
    }
};

/// Save the results file in TSV format. One line for each "point"; line length is the number of "rolls".
void saveToFile(std::string filename, float const* buffer, Box const& box)
{
    std::ofstream output(filename);
    std::cout << "Writing " << filename << std::endl;
    auto const lineLength = box.extentResult[0] / box.extentRand[0];
    for(Box::Idx i = 0; i < box.extentResult[0]; ++i)
    {
        output << buffer[i] << ((i + 1) % lineLength ? "\t" : "\n");
    }
    output.close();
}

template<Strategy TStrategy>
void runStrategy(Box& box)
{
    // Set up the pointer to the PRNG states buffer
    alpaka::rand::Philox4x32x10<Box::Acc>* const ptrBufAccRand{alpaka::getPtrNative(box.bufAccRand)};

    // Initialize the PRNG and its states on the device
    InitRandomKernel<TStrategy> initRandomKernel;
    if constexpr(TStrategy == Strategy::offset) // offset strategy needs an additional parameter for
                                                // initialisation: the offset cannot be deduced form the size of
                                                // the PRNG buffer and has to be passed in explicitly
    {
        alpaka::exec<Box::Acc>(
            box.queue,
            box.workdivRand,
            initRandomKernel,
            box.extentRand,
            ptrBufAccRand,
            box.extentResult[0] / box.extentRand[0]); // == NUM_ROLLS; amount of work to be performed by each thread
    }
    else // other strategies deduce the initial parameters solely from the thread index
    {
        alpaka::exec<Box::Acc>(box.queue, box.workdivRand, initRandomKernel, box.extentRand, ptrBufAccRand);
    }
    alpaka::wait(box.queue);

    // OPTIONAL: copy the the initial states to host if you want to check them yourself
    // alpaka_rand::Philox4x32x10<Box::Acc>* const ptrBufHostRand{alpaka::getPtrNative(box.bufHostRand)};
    // alpaka::memcpy(box.queue, box.bufHostRand, box.bufAccRand, box.extentRand);
    // alpaka::wait(box.queue);

    // Set up the pointers to the results buffers
    float* const ptrBufHostResult{alpaka::getPtrNative(box.bufHostResult)};
    float* const ptrBufAccResult{alpaka::getPtrNative(box.bufAccResult)};

    // Initialise the results buffer to zero
    for(Box::Idx i = 0; i < box.extentResult[0]; ++i)
        ptrBufHostResult[i] = 0;

    // Run the "computation" kernel filling the results buffer with random numbers in parallel
    alpaka::memcpy(box.queue, box.bufAccResult, box.bufHostResult, box.extentResult);
    FillKernel fillKernel;
    alpaka::exec<Box::Acc>(box.queue, box.workdivResult, fillKernel, box.extentResult, ptrBufAccRand, ptrBufAccResult);
    alpaka::memcpy(box.queue, box.bufHostResult, box.bufAccResult, box.extentResult);
    alpaka::wait(box.queue);

    // save the results to a CSV file
    if constexpr(TStrategy == Strategy::seed)
        saveToFile("out_seed.tsv", ptrBufHostResult, box);
    else if constexpr(TStrategy == Strategy::subsequence)
        saveToFile("out_subsequence.tsv", ptrBufHostResult, box);
    else if constexpr(TStrategy == Strategy::offset)
        saveToFile("out_offset.tsv", ptrBufHostResult, box);
    else
        std::cout << "Cannot select output filename: Unknown strategy" << std::endl;
}

auto main() -> int
{
    Box box; // Initialize the box

    runStrategy<Strategy::seed>(box); // threads start from different seeds
    runStrategy<Strategy::subsequence>(box); // threads use different subsequences
    runStrategy<Strategy::offset>(box); // threads start form an offset equal to the amount of work per thread

    return 0;
}
