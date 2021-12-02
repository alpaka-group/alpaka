/* Copyright 2020 Benjamin Worpitz, Sergei Bastrakov, Jakob Krude
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

//! This functor defines the function for which the integral is to be computed.
struct Function
{
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param x The argument.
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, float const x) -> float
    {
        return alpaka::math::sqrt(acc, (1.0f - x * x));
    }
};

//! The kernel executing the parallel logic.
//! Each Thread generates X pseudo random numbers and compares them with the given function.
//! The local result will be added to a global result.
struct Kernel
{
    //! The kernel entry point.
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam TFunctor A wrapper for a function.
    //! \param acc The accelerator to be executed on.
    //! \param numPoints The total number of points to be calculated.
    //! \param globalCounter The sum of all local results.
    //! \param functor The function for which the integral is to be computed.
    template<typename TAcc, typename TFunctor>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        size_t const num_points,
        uint32_t* const global_counter,
        TFunctor functor) const -> void
    {
        // Get the global linearized thread idx.
        auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const global_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearized_global_thread_idx = alpaka::mapIdx<1u>(global_thread_idx, global_thread_extent)[0];
        // Setup generator engine and distribution.
        auto engine = alpaka::rand::engine::createDefault(
            acc,
            linearized_global_thread_idx,
            0); // No specific subsequence start.
        // For simplicity the interval is fixed to [0.0,1.0].
        auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));

        uint32_t local_count = 0;
        for(size_t i = linearized_global_thread_idx; i < num_points; i += global_thread_extent.prod())
        {
            // Generate a point in the 2D interval.
            float x = dist(engine);
            float y = dist(engine);
            // Count every time where the point is "below" the given function.
            if(y <= functor(acc, x))
            {
                ++local_count;
            }
        }

        // Add the local result to the sum of the other results.
        alpaka::atomicAdd(acc, global_counter, local_count, alpaka::hierarchy::Blocks{});
    }
};


auto main() -> int
{
    // Defines and setup.
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::DevCpu;
    auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);
    auto const dev_host = alpaka::getDevByIdx<Host>(0u);
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{dev_acc};

    using BufHost = alpaka::Buf<Host, uint32_t, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, uint32_t, Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    // Problem parameter.
    constexpr size_t num_points = 100000000u;
    constexpr size_t extent = 1u;
    constexpr size_t num_threads = 100u; // Kernel will decide numCalcPerThread.
    constexpr size_t num_alpaka_elements_per_thread = 1;
    WorkDiv workdiv{alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        Vec(num_threads),
        Vec(num_alpaka_elements_per_thread),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    // Setup buffer.
    BufHost buf_host{alpaka::allocBuf<uint32_t, Idx>(dev_host, extent)};
    uint32_t* const ptr_buf_host{alpaka::getPtrNative(buf_host)};
    BufAcc buf_acc{alpaka::allocBuf<uint32_t, Idx>(dev_acc, extent)};
    uint32_t* const ptr_buf_acc{alpaka::getPtrNative(buf_acc)};

    // Initialize the global count to 0.
    ptr_buf_host[0] = 0.0f;
    alpaka::memcpy(queue, buf_acc, buf_host, extent);

    Kernel kernel;
    alpaka::exec<Acc>(queue, workdiv, kernel, num_points, ptr_buf_acc, Function{});
    alpaka::memcpy(queue, buf_host, buf_acc, extent);
    alpaka::wait(queue);

    // Check the result.
    uint32_t global_count = *ptr_buf_host;

    // Final result.
    float final_result = global_count / static_cast<float>(num_points);
    constexpr double pi = 3.14159265358979323846;
    constexpr double exact_result = pi / 4.0;
    auto const error = std::abs(final_result - exact_result);

    std::cout << "exact result (pi / 4): " << pi / 4.0 << "\n";
    std::cout << "final result: " << final_result << "\n";
    std::cout << "error: " << error << "\n";
    return error > 0.001 ? EXIT_FAILURE : EXIT_SUCCESS;
}
