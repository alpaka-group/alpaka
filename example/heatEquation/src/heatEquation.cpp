/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude,
 *                Sergei Bastrakov
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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>


//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x
//!
//! \param uCurrBuf grid values of u for each x and the current value of t:
//!                 u(x, t) | t = t_current
//! \param uNext resulting grid values of u for each x and the next value of t:
//!              u(x, t) | t = t_current + dt
//! \param extent number of grid nodes in x (eq. to numNodesX)
//! \param dx step in x
//! \param dt step in t

struct HeatEquationKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double const* const u_curr_buf,
        double* const u_next_buf,
        uint32_t const extent,
        double const dx,
        double const dt) const -> void
    {
        // Each kernel executes one element
        double const r = dt / (dx * dx);
        int idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if(idx > 0 && idx < extent - 1u)
        {
            u_next_buf[idx] = u_curr_buf[idx] * (1.0 - 2.0 * r) + u_curr_buf[idx - 1] * r + u_curr_buf[idx + 1] * r;
        }
    }
};


//! Exact solution to the test problem
//! u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
//! u(0, t) = u(1, t) = 0
//! u(x, 0) = sin(pi * x)
//!
//! \param x value of x
//! \param t value of t
double exact_solution(double const x, double const t)
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-pi * pi * t) * std::sin(pi * x);
}


//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
auto main() -> int
{
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else
    // Parameters (a user is supposed to change numNodesX, numTimeSteps)
    uint32_t const num_nodes_x = 1000;
    uint32_t const num_time_steps = 10000;
    double const t_max = 0.001;
    // x in [0, 1], t in [0, tMax]
    double const dx = 1.0 / static_cast<double>(num_nodes_x - 1);
    double const dt = t_max / static_cast<double>(num_time_steps - 1);

    // Check the stability condition
    double const r = dt / (dx * dx);
    if(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Set Dim and Idx type
    using Dim = alpaka::DimInt<1u>;
    using Idx = uint32_t;

    // Select accelerator-types for host and device
    // using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    using DevHost = alpaka::DevCpu;

    // Select specific devices
    auto const dev_acc = alpaka::getDevByIdx<Acc>(0u);
    auto const dev_host = alpaka::getDevByIdx<DevHost>(0u);

    // Get valid workdiv for the given problem
    uint32_t elem_per_thread = 1;
    alpaka::Vec<Dim, Idx> const extent{num_nodes_x};
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    auto workdiv = WorkDiv{alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        extent,
        elem_per_thread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};

    // Select queue
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{dev_acc};

    // Initialize host-buffer
    using BufHost = alpaka::Buf<DevHost, double, Dim, Idx>;
    // This buffer holds the calculated values
    auto u_next_buf_host = BufHost{alpaka::allocBuf<double, Idx>(dev_host, extent)};
    // This buffer will hold the current values (used for the next step)
    auto u_curr_buf_host = BufHost{alpaka::allocBuf<double, Idx>(dev_host, extent)};

    double* const p_curr_host = alpaka::getPtrNative(u_curr_buf_host);
    double* const p_next_host = alpaka::getPtrNative(u_next_buf_host);

    // Accelerator buffer
    using BufAcc = alpaka::Buf<Acc, double, Dim, Idx>;
    auto u_next_buf_acc = BufAcc{alpaka::allocBuf<double, Idx>(dev_acc, extent)};
    auto u_curr_buf_acc = BufAcc{alpaka::allocBuf<double, Idx>(dev_acc, extent)};

    double* p_curr_acc = alpaka::getPtrNative(u_curr_buf_acc);
    double* p_next_acc = alpaka::getPtrNative(u_next_buf_acc);

    // Apply initial conditions for the test problem
    for(uint32_t i = 0; i < num_nodes_x; i++)
    {
        p_curr_host[i] = exact_solution(i * dx, 0.0);
    }

    HeatEquationKernel kernel;

    // Copy host -> device
    alpaka::memcpy(queue, u_curr_buf_acc, u_curr_buf_host, extent);
    // Copy to the buffer for next as well to have boundary values set
    alpaka::memcpy(queue, u_next_buf_acc, u_curr_buf_acc, extent);
    alpaka::wait(queue);

    for(uint32_t step = 0; step < num_time_steps; step++)
    {
        // Compute next values
        alpaka::exec<Acc>(queue, workdiv, kernel, p_curr_acc, p_next_acc, num_nodes_x, dx, dt);

        // We assume the boundary conditions are constant and so these values
        // do not need to be updated.
        // So we just swap next to curr (shallow copy)
        std::swap(p_curr_acc, p_next_acc);
    }

    // Copy device -> host
    alpaka::memcpy(queue, u_next_buf_host, u_next_buf_acc, extent);
    alpaka::wait(queue);

    // Calculate error
    double max_error = 0.0;
    for(uint32_t i = 0; i < num_nodes_x; i++)
    {
        auto const error = std::abs(p_next_host[i] - exact_solution(i * dx, t_max));
        max_error = std::max(max_error, error);
    }

    double const error_threshold = 1e-5;
    bool result_correct = (max_error < error_threshold);
    if(result_correct)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: error = " << max_error << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
#endif
}
