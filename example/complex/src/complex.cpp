/* Copyright 2023 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <cstdint>
#include <iostream>

//! Complex numbers demonstration kernel
struct ComplexKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        // alpaka::Complex<T> supports the same methods as std::complex<T>, they are also useable inside kernels
        auto x = alpaka::Complex<float>(0.1f, 0.2f);
        float const real = x.real();
        auto y = alpaka::Complex<float>(0.3f, 0.4f);

        // Operators are also the same
        x *= 2.0f;
        alpaka::Complex<float> z = x + y;

        // In-kernel math functions are accessed via alpaka wrappers, the same way as for real numbers
        float zAbs = alpaka::math::abs(acc, z);
    }
};

auto main() -> int
{
    using Idx = std::size_t;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    //
    // Each accelerator has strengths and weaknesses. Therefore,
    // they need to be choosen carefully depending on the actual
    // use case. Furthermore, some accelerators only support a
    // particular workdiv, but workdiv can also be generated
    // automatically.

    // By exchanging the Acc and Queue types you can select where to execute the kernel.
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    Idx const threadsPerGrid = 1u;
    Idx const elementsPerThread = 1u;

    ComplexKernel complexKernel;

    auto const& bundeledKernel = alpaka::makeKernelBundle<Acc>(complexKernel);
    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = alpaka::getValidWorkDivForKernel(devAcc, bundeledKernel, threadsPerGrid, elementsPerThread);

    // Run the kernel
    alpaka::exec<Acc>(queue, workDiv, complexKernel);
    alpaka::wait(queue);

    // Usage of alpaka::Complex<T> on the host side is the same as inside kernels, except math functions are not
    // supported
    auto x = alpaka::Complex<float>(0.1f, 0.2f);
    float const real = x.real();
    auto y = alpaka::Complex<float>(0.3f, 0.4f);
    x *= 2.0f;
    alpaka::Complex<float> z = x + y;

    return EXIT_SUCCESS;
}
