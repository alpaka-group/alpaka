/* Copyright 2019 Benjamin Worpitz, Matthias Werner
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

#include <random>
#include <iostream>
#include <typeinfo>

// Function we are integrating on x in [0, 1]
// for alpaka version will be ALPAKA_FN_ACC
float f(float x)
{
    // Note: in alpaka it's alpaka::math::sqrt(acc, x);
    return sqrt(1.0f - x * x);
}

// Integrate using numPoints points, we assume x and f(x) belong to [0, 1]
// Will become alpaka kernel
float integrate(uint32_t numPoints)
{
    uint32_t count = 0;
    // In alpaka this loop will be parallelized with threads
    // I think for this type of application it's best to write a kernel so that
    // it works for any work division.
    // So not assume one point per thread, but have a loop in kernel in form of
    // for (i = global_thread_idx; i < numPoints; i += global_num_threads)
    for (uint32_t i = 0u; i < numPoints; i++)
    {
        // Generate a random point (x, y) in our area, [0, 1] x [0, 1]
        // in kernel will use alpaka's random generator
        // we need to take care the seeds for different threads are chosen properly (see the docs or I can help)
        float x = (float)rand() / (float)RAND_MAX;
        float y = (float)rand() / (float)RAND_MAX;
        if (y <= f(x))
            ++count;
    }
    // In alpaka here we will reduce local variables count into the shared and then global one
    // using atomic operations

    // This ratio is the integral value divided by the area square, the latter is 1 with out assumptions
    // So the ratio is just the integral value
    return (float)count / (float)numPoints;
}


auto main()
-> int
{
    uint32_t numPoints = 10000u;
    auto result = integrate(numPoints);
    std::cout << "result = " << result << ", expected = " << 3.14f / 4.0f << "\n";
    return 0;
}
