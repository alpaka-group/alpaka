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
#include <tuple>

unsigned constexpr num_calculations = 256;
unsigned constexpr num_x = 1237;
unsigned constexpr num_y = 2131;

/// Selected PRNG engine for single-value operation
template<typename TAcc>
using RandomEngineSingle = alpaka::rand::Philox4x32x10<TAcc>;
// using RandomEngineSingle = alpaka::rand::engine::uniform_cuda_hip::Xor;
// using RandomEngineSingle = alpaka::rand::engine::cpu::MersenneTwister;
// using RandomEngineSingle = alpaka::rand::engine::cpu::TinyMersenneTwister;


/// Selected PRNG engine for vector operation
template<typename TAcc>
using RandomEngineVector = alpaka::rand::Philox4x32x10Vector<TAcc>;

struct InitRandomKernel
{
    template<typename TAcc, typename TExtent, typename TRandEngine>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        TRandEngine* const states,
        std::size_t pitch_rand) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const linear_idx = alpaka::mapIdx<1u>(idx, extent)[0];
        auto const memory_location_idx = idx[0] * pitch_rand + idx[1];
        TRandEngine engine(42, static_cast<std::uint32_t>(linear_idx));
        states[memory_location_idx] = engine;
    }
};

struct RunTimestepKernelSingle
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        RandomEngineSingle<TAcc>* const states,
        const float* const cells,
        std::size_t pitch_rand,
        std::size_t pitch_out) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const memory_location_rand_idx = idx[0] * pitch_rand + idx[1];
        auto const memory_location_out_idx = idx[0] * pitch_out + idx[1];

        // Setup generator and distribution.
        RandomEngineSingle<TAcc> engine(states[memory_location_rand_idx]);
        alpaka::rand::UniformReal<float> dist;

        float sum = 0;
        for(unsigned num_calculations = 0; num_calculations < num_calculations; ++num_calculations)
        {
            sum += dist(engine);
        }
        cells[memory_location_out_idx] = sum / num_calculations;
        states[memory_location_rand_idx] = engine;
    }
};

struct RunTimestepKernelVector
{
    template<typename TAcc, typename TExtent>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TExtent const extent,
        RandomEngineVector<TAcc>* const states,
        const float* const cells,
        std::size_t pitch_rand,
        std::size_t pitch_out) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const memory_location_rand_idx = idx[0] * pitch_rand + idx[1];
        auto const memory_location_out_idx = idx[0] * pitch_out + idx[1];

        // Setup generator and distribution.
        RandomEngineVector<TAcc> engine(states[memory_location_rand_idx]); // Load the state of the random engine
        using DistributionResult =
            typename RandomEngineVector<TAcc>::template ResultContainer<float>; // Container type which will store the
                                                                                // distribution results
        unsigned constexpr result_vector_size = std::tuple_size<DistributionResult>::value; // Size of the result vector
        alpaka::rand::UniformReal<DistributionResult> dist; // Vector-aware distribution function


        float sum = 0;
        static_assert(
            num_calculations % result_vector_size == 0,
            "Number of calculations must be a multiple of result vector size.");
        for(unsigned num_calculations = 0; num_calculations < num_calculations / result_vector_size; ++num_calculations)
        {
            auto result = dist(engine);
            for(unsigned i = 0; i < result_vector_size; ++i)
            {
                sum += result[i];
            }
        }
        cells[memory_location_out_idx] = sum / num_calculations;
        states[memory_location_rand_idx] = engine;
    }
};

auto main() -> int
{
    using Dim = alpaka::DimInt<2>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);
    auto const dev_host = alpaka::getDevByIdx<Host>(0u);
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{dev_acc};

    using BufHost = alpaka::Buf<Host, float, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, float, Dim, Idx>;
    using BufHostRand = alpaka::Buf<Host, RandomEngineSingle<Acc>, Dim, Idx>;
    using BufAccRand = alpaka::Buf<Acc, RandomEngineSingle<Acc>, Dim, Idx>;
    using BufHostRandVec = alpaka::Buf<Host, RandomEngineVector<Acc>, Dim, Idx>;
    using BufAccRandVec = alpaka::Buf<Acc, RandomEngineVector<Acc>, Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    constexpr Idx num_x = num_x;
    constexpr Idx num_y = num_y;

    const Vec extent(num_y, num_x);

    constexpr Idx per_thread_x = 1;
    constexpr Idx per_thread_y = 1;

    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        extent,
        Vec(per_thread_y, per_thread_x),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    // Setup buffer.
    BufHost buf_host_s{alpaka::allocBuf<float, Idx>(dev_host, extent)};
    float* const ptr_buf_host_s{alpaka::getPtrNative(buf_host_s)};
    BufAcc buf_acc_s{alpaka::allocBuf<float, Idx>(dev_acc, extent)};
    float* const ptr_buf_acc_s{alpaka::getPtrNative(buf_acc_s)};

    BufHost buf_host_v{alpaka::allocBuf<float, Idx>(dev_host, extent)};
    float* const ptr_buf_host_v{alpaka::getPtrNative(buf_host_v)};
    BufAcc buf_acc_v{alpaka::allocBuf<float, Idx>(dev_acc, extent)};
    float* const ptr_buf_acc_v{alpaka::getPtrNative(buf_acc_v)};

    BufHostRand buf_host_rand_s{alpaka::allocBuf<RandomEngineSingle<Acc>, Idx>(dev_host, extent)};
    BufAccRand buf_acc_rand_s{alpaka::allocBuf<RandomEngineSingle<Acc>, Idx>(dev_acc, extent)};
    RandomEngineSingle<Acc>* const ptr_buf_acc_rand_s{alpaka::getPtrNative(buf_acc_rand_s)};

    BufHostRandVec buf_host_rand_v{alpaka::allocBuf<RandomEngineVector<Acc>, Idx>(dev_host, extent)};
    BufAccRandVec buf_acc_rand_v{alpaka::allocBuf<RandomEngineVector<Acc>, Idx>(dev_acc, extent)};
    RandomEngineVector<Acc>* const ptr_buf_acc_rand_v{alpaka::getPtrNative(buf_acc_rand_v)};

    InitRandomKernel init_random_kernel;
    auto pitch_buf_acc_rand_s = alpaka::getPitchBytes<1u>(buf_acc_rand_s) / sizeof(RandomEngineSingle<Acc>);
    alpaka::exec<Acc>(queue, workdiv, init_random_kernel, extent, ptr_buf_acc_rand_s, pitch_buf_acc_rand_s);
    alpaka::wait(queue);

    auto pitch_buf_acc_rand_v = alpaka::getPitchBytes<1u>(buf_acc_rand_v) / sizeof(RandomEngineVector<Acc>);
    alpaka::exec<Acc>(queue, workdiv, init_random_kernel, extent, ptr_buf_acc_rand_v, pitch_buf_acc_rand_v);
    alpaka::wait(queue);

    auto pitch_host_s = alpaka::getPitchBytes<1u>(buf_host_s) / sizeof(float); /// \todo: get the type from bufHostS
    auto pitch_host_v = alpaka::getPitchBytes<1u>(buf_host_v) / sizeof(float); /// \todo: get the type from bufHostV

    for(Idx y = 0; y < num_y; ++y)
    {
        for(Idx x = 0; x < num_x; ++x)
        {
            ptr_buf_host_s[y * pitch_host_s + x] = 0;
            ptr_buf_host_v[y * pitch_host_v + x] = 0;
        }
    }

    /// \todo get the types from respective function parameters
    auto pitch_buf_acc_s = alpaka::getPitchBytes<1u>(buf_acc_s) / sizeof(float);
    alpaka::memcpy(queue, buf_acc_s, buf_host_s, extent);
    RunTimestepKernelSingle run_timestep_kernel_single;
    alpaka::exec<Acc>(
        queue,
        workdiv,
        run_timestep_kernel_single,
        extent,
        ptr_buf_acc_rand_s,
        ptr_buf_acc_s,
        pitch_buf_acc_rand_s,
        pitch_buf_acc_s);
    alpaka::memcpy(queue, buf_host_s, buf_acc_s, extent);

    auto pitch_buf_acc_v = alpaka::getPitchBytes<1u>(buf_acc_v) / sizeof(float);
    alpaka::memcpy(queue, buf_acc_v, buf_host_v, extent);
    RunTimestepKernelVector run_timestep_kernel_vector;
    alpaka::exec<Acc>(
        queue,
        workdiv,
        run_timestep_kernel_vector,
        extent,
        ptr_buf_acc_rand_v,
        ptr_buf_acc_v,
        pitch_buf_acc_rand_v,
        pitch_buf_acc_v);
    alpaka::memcpy(queue, buf_host_v, buf_acc_v, extent);
    alpaka::wait(queue);

    float avg_s = 0;
    float avg_v = 0;
    for(Idx y = 0; y < num_y; ++y)
    {
        for(Idx x = 0; x < num_x; ++x)
        {
            avg_s += ptr_buf_host_s[y * pitch_host_s + x];
            avg_v += ptr_buf_host_v[y * pitch_host_v + x];
        }
    }
    avg_s /= num_x * num_y;
    avg_v /= num_x * num_y;

    std::cout << "Number of cells: " << num_x * num_y << "\n";
    std::cout << "Number of calculations: " << num_calculations << "\n";
    std::cout << "Mean value A: " << avg_s << " (should converge to 0.5)\n";
    std::cout << "Mean value B: " << avg_v << " (should converge to 0.5)\n";

    return 0;
}
