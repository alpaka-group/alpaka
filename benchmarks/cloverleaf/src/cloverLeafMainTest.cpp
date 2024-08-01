#include "cloverLeafKernels.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <experimental/mdspan>
#include <iostream>

/**
 * Cloverleaf benchmark
 */

//! \brief The Function for testing cloverleaf kernels for given Acc type and data type.
//! \tparam Acc the accelerator type
//! \tparam DataType The data type to differentiate single or double data type based tests.
template<typename Acc, typename DataType>
void testKernels()
{
    // Define device and queue
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    Queue queue(devAcc);

    // Define the 2D extent (dimensions)
    alpaka::Vec<Dim2, Idx> const extent2D(static_cast<Idx>(nx), static_cast<Idx>(ny));

    // Allocate host memory
    auto hDensity = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hEnergy = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hPressure = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hVelocityX = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hVelocityY = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hSoundspeed = alpaka::allocBuf<DataType, Idx>(devHost, extent2D); // Additional buffer for soundspeed

    // Allocate device memory
    auto dDensity = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dEnergy = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dPressure = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dVelocityX = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dVelocityY = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dSoundspeed = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D); // Additional buffer for soundspeed

    // Convert directly to mdspan
    auto mdDensity = alpaka::experimental::getMdSpan(dDensity);
    auto mdEnergy = alpaka::experimental::getMdSpan(dEnergy);
    auto mdPressure = alpaka::experimental::getMdSpan(dPressure);
    auto mdVelocityX = alpaka::experimental::getMdSpan(dVelocityX);
    auto mdVelocityY = alpaka::experimental::getMdSpan(dVelocityY);
    auto mdSoundspeed = alpaka::experimental::getMdSpan(dSoundspeed); // Additional mdspan for soundspeed

    auto hFluxDensity = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hFluxEnergy = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hFluxVelocityX = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);
    auto hFluxVelocityY = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);

    auto dFluxDensity = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dFluxEnergy = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dFluxVelocityX = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);
    auto dFluxVelocityY = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);

    auto mdFluxDensity = alpaka::experimental::getMdSpan(dFluxDensity);
    auto mdFluxEnergy = alpaka::experimental::getMdSpan(dFluxEnergy);
    auto mdFluxVelocityX = alpaka::experimental::getMdSpan(dFluxVelocityX);
    auto mdFluxVelocityY = alpaka::experimental::getMdSpan(dFluxVelocityY);

    // Allocate host memory for viscosity
    auto hViscosity = alpaka::allocBuf<DataType, Idx>(devHost, extent2D);

    // Allocate device memory for viscosity
    auto dViscosity = alpaka::allocBuf<DataType, Idx>(devAcc, extent2D);

    // Create mdspan view for viscosity using alpaka::experimental::getMdSpan
    auto mdViscosity = alpaka::experimental::getMdSpan(dViscosity);

    // Define work divisions
    const alpaka::Vec<Dim2, Idx> size{nx, ny};

    auto const bundledInitKernel
        = alpaka::KernelBundle(InitializerKernel{}, mdDensity, mdEnergy, mdPressure, mdVelocityX, mdVelocityY);

    auto workDiv = alpaka::getValidWorkDivForKernel<Acc>(
        devAcc,
        bundledInitKernel,
        size,
        alpaka::Vec<Dim2, Idx>::ones(),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    // Simulation parameters
    float totalTime = 0.0f; // Total simulation time
    float endTime = 16.0f; // End time of the simulation

    // Exec the initialization kernel
    alpaka::exec<Acc>(queue, workDiv, InitializerKernel{}, mdDensity, mdEnergy, mdPressure, mdVelocityX, mdVelocityY);

    std::cout << "InitializerKernel finished" << std::endl;

    // Wait for the kernel to finish
    alpaka::wait(queue);

    // Variables to calculate the dt at each step
    float dx = 10.0f / nx;
    float dy = 10.0f / ny;
    float cMax = 0.5f;
    float dt = 0.04f;

    alpaka::Vec<alpaka::DimInt<1u>, Idx> const extent1D = alpaka::Vec<alpaka::DimInt<1u>, Idx>::all(1u);
    auto resultGpuHost = alpaka::allocBuf<DataType, Idx>(devHost, extent1D);
    resultGpuHost[0] = static_cast<DataType>(1.0f);
    auto outputDeviceMemory = alpaka::allocBuf<DataType, Idx>(devAcc, extent1D);
    alpaka::memcpy(queue, outputDeviceMemory, resultGpuHost);
    std::cout << "Workdiv:" << workDiv << std::endl;
    std::cout << "Running cloverleaf loop." << std::endl;

    int iteration = 0;

    while(totalTime < endTime)
    {
        // Calculate the time step (dt), max velocity is needed.
        // Launch the kernel to compute the maximum velocity
        alpaka::exec<Acc>(
            queue,
            workDiv,
            MaxVelocityKernel{},
            mdVelocityX,
            mdVelocityY,
            std::data(outputDeviceMemory));

        // Copy result from device memory to host
        alpaka::memcpy(queue, resultGpuHost, outputDeviceMemory, extent1D);

        // std::cout << "MaxVelocityKernel finished" << std::endl;

        alpaka::wait(queue);

        // Now use maxVelocity to compute the time step
        dt = calculateTimeStep(dx, dy, resultGpuHost[0], cMax);

        // Exec the halo update kernel
        alpaka::exec<Acc>(queue, workDiv, UpdateHaloKernel{}, mdDensity);

        // std::cout << "Halo kernel finished" << std::endl;

        // Wait for the halo update kernel to finish
        alpaka::wait(queue);

        // Exec the ideal gas kernel
        alpaka::exec<Acc>(
            queue,
            workDiv,
            IdealGasKernel{},
            mdDensity,
            mdEnergy,
            mdPressure,
            mdSoundspeed,
            1.4f); // gamma

        // std::cout << "IdealGasKernel finished" << std::endl;

        // Wait for the ideal gas kernel to finish
        alpaka::wait(queue);

        // Exec the EOS kernel
        float gamma = 1.4f; // Specific heat ratio
        alpaka::exec<
            Acc>(queue, workDiv, EOSKernel{}, mdDensity, mdEnergy, mdPressure, mdVelocityX, mdVelocityY, gamma);

        // std::cout << "EOSKernel finished" << std::endl;

        // Wait for the EOS kernel to finish
        alpaka::wait(queue);

        // Exec the Flux kernel
        alpaka::exec<Acc>(
            queue,
            workDiv,
            FluxKernel{},
            mdDensity,
            mdEnergy,
            mdPressure,
            mdVelocityX,
            mdVelocityY,
            mdFluxDensity,
            mdFluxEnergy,
            mdFluxVelocityX,
            mdFluxVelocityY);

        // std::cout << "FluxKernel finished" << std::endl;

        // Wait for the Flux kernel to finish
        alpaka::wait(queue);

        // Exec the advection kernel (fourth step)
        alpaka::exec<Acc>(queue, workDiv, AdvectionKernel{}, mdDensity, mdVelocityX, mdVelocityY);

        // std::cout << "AdvectionKernel finished" << std::endl;

        // Wait for the Advection kernel to finish
        alpaka::wait(queue);

        // Copy data back to host for verification (optional)
        alpaka::memcpy(queue, hDensity, dDensity);
        alpaka::memcpy(queue, hEnergy, dEnergy);
        alpaka::memcpy(queue, hPressure, dPressure);
        alpaka::memcpy(queue, hVelocityX, dVelocityX);
        alpaka::memcpy(queue, hVelocityY, dVelocityY);

        // Flux densities
        alpaka::memcpy(queue, hFluxDensity, dFluxDensity);
        alpaka::memcpy(queue, hFluxEnergy, dFluxEnergy);
        alpaka::memcpy(queue, hFluxVelocityX, dFluxVelocityX);
        alpaka::memcpy(queue, hFluxVelocityY, dFluxVelocityY);

        // Wait for data transfer to complete
        alpaka::wait(queue);

        // Exec the Lagrangian kernel
        alpaka::exec<Acc>(
            queue,
            workDiv,
            LagrangianKernel{},
            mdDensity,
            mdEnergy,
            mdVelocityX,
            mdVelocityY,
            mdFluxDensity,
            mdFluxEnergy,
            mdFluxVelocityX,
            mdFluxVelocityY);

        // std::cout << "LagrangianKernel finished" << std::endl;

        // Wait for the Lagrangian kernel to finish
        alpaka::wait(queue);

        // Exec the Viscosity kernel
        alpaka::exec<
            Acc>(queue, workDiv, ViscosityKernel{}, mdDensity, mdVelocityX, mdVelocityY, mdPressure, mdViscosity);

        // std::cout << "ViscosityKernel finished" << std::endl;

        // Wait for the Viscosity kernel to finish
        alpaka::wait(queue);

        // Update the simulation time
        totalTime += dt;

        if(iteration % 2000 == 0)
        {
            std::cout << "Current Time: " << totalTime << ", End Time: " << endTime << ", Step Size: " << dt
                      << ", iteration: " << iteration << std::endl;
        }

        iteration++;
    }

    std::cout << "Copying results back." << std::endl;

    // Copy final data back to host for verification
    alpaka::memcpy(queue, hDensity, dDensity);
    alpaka::memcpy(queue, hEnergy, dEnergy);
    alpaka::memcpy(queue, hPressure, dPressure);
    alpaka::memcpy(queue, hVelocityX, dVelocityX);
    alpaka::memcpy(queue, hVelocityY, dVelocityY);
    alpaka::memcpy(queue, hFluxDensity, dFluxDensity);
    alpaka::memcpy(queue, hFluxEnergy, dFluxEnergy);
    alpaka::memcpy(queue, hFluxVelocityX, dFluxVelocityX);
    alpaka::memcpy(queue, hFluxVelocityY, dFluxVelocityY);
    alpaka::memcpy(queue, hViscosity, dViscosity);

    std::cout << "Data copy to host finished" << std::endl;

    // Wait for all data transfers to complete
    alpaka::wait(queue);

    // Verification (check for NaNs and Infs)
    auto mdHostDensity = alpaka::experimental::getMdSpan(hDensity);
    auto mdHostEnergy = alpaka::experimental::getMdSpan(hEnergy);
    auto mdHostPressure = alpaka::experimental::getMdSpan(hPressure);
    auto mdHostVelocityX = alpaka::experimental::getMdSpan(hVelocityX);
    auto mdHostVelocityY = alpaka::experimental::getMdSpan(hVelocityY);

    bool success = true;
    for(Idx i = 0; i < nx; ++i)
    {
        for(Idx j = 0; j < ny; ++j)
        {
            if(std::isnan(mdHostDensity(i, j)) || std::isinf(mdHostDensity(i, j)) || std::isnan(mdHostEnergy(i, j))
               || std::isinf(mdHostEnergy(i, j)) || std::isnan(mdHostPressure(i, j))
               || std::isinf(mdHostPressure(i, j)) || std::isnan(mdHostVelocityX(i, j))
               || std::isinf(mdHostVelocityX(i, j)) || std::isnan(mdHostVelocityY(i, j))
               || std::isinf(mdHostVelocityY(i, j)))
            {
                success = false;
                break;
            }
        }
    }

    if(success)
    {
        std::cout << "Simulation succeeded!" << std::endl;
    }
    else
    {
        std::cout << "Simulation failed due to NaN or Inf values!" << std::endl;
    }
}

using TestAccs2D = alpaka::test::EnabledAccs<alpaka::DimInt<2u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("TEST: Cloverleaf", "[benchmark-test]", TestAccs2D)
{
    using Acc = TestType;
    if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt>)
    {
        // Run tests for float data type
        testKernels<Acc, float>();
    }
}
