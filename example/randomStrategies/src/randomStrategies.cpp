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
#include <fstream>
#include <iostream>

// This example generates NUM_ROLLS of random events for each of NUM_POINTS points.
unsigned constexpr num_points = 2000; ///< Number of "points". Each will be  processed by a single thread.
unsigned constexpr num_rolls = 2000; ///< Amount of random number "dice rolls" performed for each "point".

/// Selected PRNG engine
// Comment the current "using" line, and uncomment a different one to change the PRNG engine
template<typename TAcc>
using RandomEngine = alpaka::rand::Philox4x32x10<TAcc>;
// using RandomEngine = alpaka::rand::engine::cpu::MersenneTwister;
// using RandomEngine = alpaka::rand::engine::cpu::TinyMersenneTwister;
// using RandomEngine = alpaka::rand::engine::uniform_cuda_hip::Xor;


/// Parameters to set up the default accelerator, queue, and buffers
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

    QueueAcc m_queue; ///< default accelerator queue

    // buffers holding the PRNG states
    using BufHostRand = alpaka::Buf<Host, RandomEngine<Acc>, Dim, Idx>;
    using BufAccRand = alpaka::Buf<Acc, RandomEngine<Acc>, Dim, Idx>;

    Vec const m_extent_rand; ///< size of the buffer of PRNG states
    WorkDiv m_workdiv_rand; ///< work division for PRNG buffer initialization
    BufHostRand m_buf_host_rand; ///< host side PRNG states buffer (can be used to check the state of the states)
    BufAccRand m_buf_acc_rand; ///< device side PRNG states buffer

    // buffers holding the "simulation" results
    using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;

    Vec const m_extent_result; ///< size of the results buffer
    WorkDiv m_workdiv_result; ///< work division of the result calculation
    BufHost m_buf_host_result; ///< host side results buffer
    BufAcc m_buf_acc_result; ///< device side results buffer

    Box()
        : m_queue{alpaka::getDevByIdx<Acc>(Idx{0})}
        , m_extent_rand{static_cast<Idx>(num_points)} // One PRNG state per "point".
        , m_workdiv_rand{alpaka::getValidWorkDiv<Acc>(
              alpaka::getDevByIdx<Acc>(Idx{0}),
              m_extent_rand,
              Vec(Idx{1}),
              false,
              alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)}
        , m_buf_host_rand{alpaka::allocBuf<RandomEngine<Acc>, Idx>(alpaka::getDevByIdx<Host>(0u), m_extent_rand)}
        , m_buf_acc_rand{alpaka::allocBuf<RandomEngine<Acc>, Idx>(alpaka::getDevByIdx<Acc>(0u), m_extent_rand)}
        , m_extent_result{static_cast<Idx>((num_points * num_rolls))} // Store all "rolls" for each "point"
        , m_workdiv_result{alpaka::getValidWorkDiv<Acc>(
              alpaka::getDevByIdx<Acc>(Idx{0}),
              m_extent_result,
              Vec(static_cast<Idx>(num_rolls)), // One thread per "point"; each performs NUM_ROLLS "rolls"
              false,
              alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)}
        , m_buf_host_result{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Host>(0u), m_extent_result)}
        , m_buf_acc_result{alpaka::allocBuf<float, Idx>(alpaka::getDevByIdx<Acc>(0u), m_extent_result)}
    {
    }
};

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
};

template<>
struct InitRandomKernel<Strategy::seed>
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the PRNG states buffer
        TRandEngine* const states, ///< PRNG states buffer
        unsigned const skip_length = 0 ///< number of PRNG elements to skip (offset strategy only)
    ) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; ///< index of the current thread
        TRandEngine engine(idx, 0, 0); // Initialize the engine
        states[idx] = engine; // Save the initial state
    }
};

template<>
struct InitRandomKernel<Strategy::subsequence>
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the PRNG states buffer
        TRandEngine* const states, ///< PRNG states buffer
        unsigned const skip_length = 0 ///< number of PRNG elements to skip (offset strategy only)
    ) const -> void
    {
        alpaka::ignore_unused(skip_length);
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; ///< index of the current thread
        TRandEngine engine(0, idx, 0); // Initialize the engine
        states[idx] = engine; // Save the initial state
    }
};

template<>
struct InitRandomKernel<Strategy::offset>
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc, ///< current accelerator
        TExtent const extent, ///< size of the PRNG states buffer
        TRandEngine* const states, ///< PRNG states buffer
        unsigned const skip_length = 0 ///< number of PRNG elements to skip (offset strategy only)
    ) const -> void
    {
        alpaka::ignore_unused(skip_length);
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; ///< index of the current thread
        TRandEngine engine(0, 0, idx * skip_length); // Initialize the engine
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
        RandomEngine<TAcc>* const states, ///< PRNG states buffer
        const float* const cells ///< results buffer
    ) const -> void
    {
        /// Index of the current thread. Each thread performs multiple "dice rolls".
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        /// Number of "dice rolls" to be performed by each thread.
        auto const length = extent[0] / alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        RandomEngine<TAcc> engine(states[idx]); // Setup the PRNG using the saved state for this thread.
        alpaka::rand::UniformReal<float> dist; // Setup the random number distribution
        for(unsigned i = 0; i < length; ++i)
        {
            cells[length * idx + i] = dist(engine); // Roll the dice!
        }
        states[idx] = engine; // Save the final PRNG state
    }
};

/** Save the results to a file and show the calculated average for quick correctness check.
 *
 *  File is in TSV format. One line for each "point"; line length is the number of "rolls".
 */
void save_data_and_show_average(std::string filename, float const* buffer, Box const& box)
{
    std::ofstream output(filename);
    std::cout << "Writing " << filename << " ... " << std::flush;
    auto const line_length = box.m_extent_result[0] / box.m_extent_rand[0];
    double average = 0;
    for(Box::Idx i = 0; i < box.m_extent_result[0]; ++i)
    {
        output << buffer[i] << (((i + 1) % line_length) != 0u ? "\t" : "\n");
        average += buffer[i];
    }
    average /= box.m_extent_result[0];
    std::cout << "average value = " << average << " (should be close to 0.5)" << std::endl;
    output.close();
}

template<Strategy TStrategy>
struct Writer;

template<>
struct Writer<Strategy::seed>
{
    void static save(float const* buffer, Box const& box)
    {
        save_data_and_show_average("out_seed.csv", buffer, box);
    }
};

template<>
struct Writer<Strategy::subsequence>
{
    void static save(float const* buffer, Box const& box)
    {
        save_data_and_show_average("out_subsequence.csv", buffer, box);
    }
};

template<>
struct Writer<Strategy::offset>
{
    void static save(float const* buffer, Box const& box)
    {
        save_data_and_show_average("out_offset.csv", buffer, box);
    }
};

template<Strategy TStrategy>
void run_strategy(Box& box)
{
    // Set up the pointer to the PRNG states buffer
    RandomEngine<Box::Acc>* const ptr_buf_acc_rand{alpaka::getPtrNative(box.m_buf_acc_rand)};

    // Initialize the PRNG and its states on the device
    InitRandomKernel<TStrategy> init_random_kernel;
    // The offset strategy needs an additional parameter for initialisation: the offset cannot be deduced form the size
    // of the PRNG buffer and has to be passed in explicitly. Other strategies ignore the last parameter, and deduce
    // the initial parameters solely from the thread index

    alpaka::exec<Box::Acc>(
        box.m_queue,
        box.m_workdiv_rand,
        init_random_kernel,
        box.m_extent_rand,
        ptr_buf_acc_rand,
        box.m_extent_result[0] / box.m_extent_rand[0]); // == NUM_ROLLS; amount of work to be performed by each thread

    alpaka::wait(box.m_queue);

    // OPTIONAL: copy the the initial states to host if you want to check them yourself
    // alpaka_rand::Philox4x32x10<Box::Acc>* const ptrBufHostRand{alpaka::getPtrNative(box.bufHostRand)};
    // alpaka::memcpy(box.queue, box.bufHostRand, box.bufAccRand, box.extentRand);
    // alpaka::wait(box.queue);

    // Set up the pointers to the results buffers
    float* const ptr_buf_host_result{alpaka::getPtrNative(box.m_buf_host_result)};
    float* const ptr_buf_acc_result{alpaka::getPtrNative(box.m_buf_acc_result)};

    // Initialise the results buffer to zero
    for(Box::Idx i = 0; i < box.m_extent_result[0]; ++i)
        ptr_buf_host_result[i] = 0;

    // Run the "computation" kernel filling the results buffer with random numbers in parallel
    alpaka::memcpy(box.m_queue, box.m_buf_acc_result, box.m_buf_host_result, box.m_extent_result);
    FillKernel fill_kernel;
    alpaka::exec<Box::Acc>(box.m_queue, box.m_workdiv_result, fill_kernel, box.m_extent_result, ptr_buf_acc_rand, ptr_buf_acc_result);
    alpaka::memcpy(box.m_queue, box.m_buf_host_result, box.m_buf_acc_result, box.m_extent_result);
    alpaka::wait(box.m_queue);

    // save the results to a CSV file
    Writer<TStrategy>::save(ptr_buf_host_result, box);
}

auto main() -> int
{
    Box box; // Initialize the box

    run_strategy<Strategy::seed>(box); // threads start from different seeds
    run_strategy<Strategy::subsequence>(box); // threads use different subsequences
    run_strategy<Strategy::offset>(box); // threads start form an offset equal to the amount of work per thread

    return 0;
}
