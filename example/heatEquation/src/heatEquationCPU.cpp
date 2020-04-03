/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file exemplifies usage of Alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
extent.m_data[0] * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <typeinfo>


//#############################################################################
//! CPU version of explicit finite-difference 1d heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x
//! This function computes one time step (@Jacob: this function will become a kernel)
//!
//! \param uCurrent grid values of u for each x and the current value of t:
//!                 u(x, t) | t = t_current
//! \param numNodesX number of grid nodes in x
//! \param dx step in x
//! \param dt step in t
//! \param uNext resulting grid values of u for each x and the next value of t:
//!              u(x, t) | t = t_current + dt
void heatEquationStep(
    double const * uCurrent,
    uint32_t const numNodesX,
    double const dx,
    double const dt,
    double * uNext)
{
    // We assume boundary values are defined by Dirichlet boundary conditions
    // and so do not need to be updated in the kernel
    double const r = dt / ( dx * dx );
    for( uint32_t i = 1u; i < numNodesX; i++ )
    {
        uNext[ i ] = uCurrent[ i ] * ( 1.0 - 2.0 * r ) + uCurrent[ i - 1 ] * r
            + uCurrent[ i + 1 ] * r;
    }
}


//! Exact solution to the test problem
//! u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
//! u(0, t) = u(1, t) = 0
//! u(x, 0) = sin(pi * x)
//!
//! \param x value of x
//! \param t value of t
double exactSolution(
    double const x,
    double const t)
{
    // @Jakob, this function is for testing correctness, does not need to be alpakafied
    constexpr double pi = 3.14159265358979323846;
    return exp( -pi * pi * t ) * sin( pi * x );
}

auto main()
-> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else

    // Parameters (a user is supposed to change numNodesX, numNodesT)
    uint32_t const numNodesX = 1000;
    uint32_t const numTimeSteps = 10000;
    double const tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    double const dx = 1.0 / static_cast< double >( numNodesX - 1 );
    double const dt = tMax / static_cast< double >( numTimeSteps - 1 );

    // Check the stability condition
    double const r = dt / ( dx * dx );
    if( r > 0.5 )
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r
            << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Buffers for current and next time steps
    double * uCurrent = new double[ numNodesX ];
    double * uNext = new double[ numNodesX ];

    // Apply initial conditions for the test problem
    for( uint32_t i = 0; i < numNodesX; i++ )
        uCurrent[ i ] = exactSolution( i * dx, 0.0 );

    // Iterate in time
    for( uint32_t step = 0; step < numTimeSteps; step++ )
    {
        // Apply boundary conditions (@Jakob i think for simplicity we may
        // skip this part in alpaka version assuming that boundary conditions do not change in time)
        uCurrent[ 0 ] = exactSolution( 0, step * dt );
        uCurrent[ numNodesX - 1 ] = exactSolution( ( numNodesX - 1 ) * dx, step * dt );

        // Compute next values
        heatEquationStep( uCurrent, numNodesX, dx, dt, uNext );

        // Swap buffers for the next step
        // (@Jakob no deep copy here on purpose, just the values of uNext need to become
        // values of uCurrent for the next time step)
        std::swap( uCurrent, uNext );
    }

    // Now uCurrent has values for t = tMax, compare to the exact solution
    // TODO @Sergei: will think on automatically checking if maxError is okay for given
    // number of nodes in x and t, for now it just needs to be << 1
    double maxError = 0.0;
    for( uint32_t i = 0; i < numNodesX; i++ )
    {
        auto const error = abs( uCurrent[ i ] - exactSolution( i * dx, tMax ) );
        maxError = std::max( maxError, error );
    }
    std::cout << "Max error to the exact solution at t = tMax: " << maxError << "\n";

    delete [] uCurrent;
    delete [] uNext;
    return EXIT_SUCCESS;
#endif
}
