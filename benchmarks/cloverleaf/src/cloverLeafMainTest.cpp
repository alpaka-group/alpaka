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

float const density1 = 0.2f;
float const energy1 = 1.0f;
float const density2 = 1.0f;
float const energy2 = 2.5f;

float const initialTimestep = 0.04f;
float const timestepRise = 1.5f;
float const maxTimestep = 0.04f;
float const endTime = 15.5f;
int const endStep = 2955;

float const xmin = 0.0f;
float const ymin = 0.0f;
float const xmax = 10.0f;
float const ymax = 10.0f;


float const dx = (xmax - xmin) / nx;
float const dy = (ymax - ymin) / ny;
float const dz = 0.01f;
float const cMax = 0.5f;

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

    // Define the 3D extent (dimensions)
    alpaka::Vec<Dim3, Idx> const extent3D(static_cast<Idx>(nx), static_cast<Idx>(ny), static_cast<Idx>(nz));

    // Allocate host memory
    auto hDensity = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hEnergy = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hPressure = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hVelocityX = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hVelocityY = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hVelocityZ = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hSoundspeed = alpaka::allocBuf<DataType, Idx>(devHost, extent3D); // Additional buffer for soundspeed

    // Allocate device memory
    auto dDensity = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dEnergy = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dPressure = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dVelocityX = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dVelocityY = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dVelocityZ = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dSoundspeed = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D); // Additional buffer for soundspeed

    // Convert directly to mdspan
    auto mdDensity = alpaka::experimental::getMdSpan(dDensity);
    auto mdEnergy = alpaka::experimental::getMdSpan(dEnergy);
    auto mdPressure = alpaka::experimental::getMdSpan(dPressure);
    auto mdVelocityX = alpaka::experimental::getMdSpan(dVelocityX);
    auto mdVelocityY = alpaka::experimental::getMdSpan(dVelocityY);
    auto mdVelocityZ = alpaka::experimental::getMdSpan(dVelocityZ);
    auto mdSoundspeed = alpaka::experimental::getMdSpan(dSoundspeed); // Additional mdspan for soundspeed

    auto hFluxDensity = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hFluxEnergy = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hFluxVelocityX = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hFluxVelocityY = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);
    auto hFluxVelocityZ = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);

    auto dFluxDensity = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dFluxEnergy = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dFluxVelocityX = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dFluxVelocityY = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);
    auto dFluxVelocityZ = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);

    auto mdFluxDensity = alpaka::experimental::getMdSpan(dFluxDensity);
    auto mdFluxEnergy = alpaka::experimental::getMdSpan(dFluxEnergy);
    auto mdFluxVelocityX = alpaka::experimental::getMdSpan(dFluxVelocityX);
    auto mdFluxVelocityY = alpaka::experimental::getMdSpan(dFluxVelocityY);
    auto mdFluxVelocityZ = alpaka::experimental::getMdSpan(dFluxVelocityZ);

    // Allocate host memory for viscosity
    auto hViscosity = alpaka::allocBuf<DataType, Idx>(devHost, extent3D);

    // Allocate device memory for viscosity
    auto dViscosity = alpaka::allocBuf<DataType, Idx>(devAcc, extent3D);

    // Create mdspan view for viscosity using alpaka::experimental::getMdSpan
    auto mdViscosity = alpaka::experimental::getMdSpan(dViscosity);

    // Define work divisions
    const alpaka::Vec<Dim3, Idx> size{nx, ny, nz};

    auto const bundeledInitKernel = alpaka::KernelBundle(
        InitializerKernel{},
        mdDensity,
        mdEnergy,
        mdPressure,
        mdVelocityX,
        mdVelocityY,
        mdVelocityZ);

    auto workDiv = alpaka::getValidWorkDivForKernel<Acc>(
        devAcc,
        bundeledInitKernel,
        size,
        alpaka::Vec<Dim3, Idx>::ones(),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
    // Simulation parameters
    float totalTime = 0.0f; // Total simulation time

    // Exec the initialization kernel
    alpaka::exec<Acc>(
        queue,
        workDiv,
        InitializerKernel{},
        mdDensity,
        mdEnergy,
        mdPressure,
        mdVelocityX,
        mdVelocityY,
        mdVelocityZ);
    // Wait for the kernel to finish
    alpaka::wait(queue);

    // Variables to calculate the dt at each step
    float dt = initialTimestep;

    alpaka::Vec<alpaka::DimInt<1u>, Idx> const extent1D = alpaka::Vec<alpaka::DimInt<1u>, Idx>::all(1u);
    auto resultGpuHost = alpaka::allocBuf<DataType, Idx>(devHost, extent1D);
    resultGpuHost[0] = static_cast<DataType>(1.0f);
    auto outputDeviceMemory = alpaka::allocBuf<DataType, Idx>(devAcc, extent1D);
    alpaka::memcpy(queue, outputDeviceMemory, resultGpuHost);

    std::cout << "Workdiv: " << workDiv << std::endl;
    std::cout << "Running cloverleaf loop." << std::endl;
    int step = 0;
    while(totalTime < endTime && step < endStep)
    {
        // Calculate the time step (dt), max velocity is needed.
        // Launch the kernel to compute the maximum velocity
        alpaka::exec<Acc>(
            queue,
            workDiv,
            MaxVelocityKernel{},
            mdVelocityX,
            mdVelocityY,
            mdVelocityZ,
            std::data(outputDeviceMemory));

        // Copy result from device memory to host
        alpaka::memcpy(queue, resultGpuHost, outputDeviceMemory, extent1D);
        alpaka::wait(queue);

        // Now use maxVelocity to compute the time step
        dt = std::min((float) maxTimestep, (float) (initialTimestep * std::pow(timestepRise, step)));

        // Exec the halo update kernel
        alpaka::exec<Acc>(queue, workDiv, UpdateHaloKernel{}, mdDensity);

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

        // Wait for the ideal gas kernel to finish
        alpaka::wait(queue);

        // Exec the EOS kernel
        float gamma = 1.4f; // Specific heat ratio
        alpaka::exec<Acc>(
            queue,
            workDiv,
            EOSKernel{},
            mdDensity,
            mdEnergy,
            mdPressure,
            mdVelocityX,
            mdVelocityY,
            mdVelocityZ,
            gamma);

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
            mdVelocityZ,
            mdFluxDensity,
            mdFluxEnergy,
            mdFluxVelocityX,
            mdFluxVelocityY,
            mdFluxVelocityZ);

        // Wait for the Flux kernel to finish
        alpaka::wait(queue);

        // Exec the advection kernel (fourth step)
        alpaka::exec<Acc>(queue, workDiv, AdvectionKernel{}, mdDensity, mdVelocityX, mdVelocityY, mdVelocityZ);

        // Wait for the Advection kernel to finish
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
            mdVelocityZ,
            mdFluxDensity,
            mdFluxEnergy,
            mdFluxVelocityX,
            mdFluxVelocityY,
            mdFluxVelocityZ);

        // Wait for the Lagrangian kernel to finish
        alpaka::wait(queue);

        // Exec the Viscosity kernel
        alpaka::exec<Acc>(
            queue,
            workDiv,
            ViscosityKernel{},
            mdDensity,
            mdVelocityX,
            mdVelocityY,
            mdVelocityZ,
            mdPressure,
            mdViscosity);

        // Wait for the Viscosity kernel to finish
        alpaka::wait(queue);

        // Update the simulation time
        totalTime += dt;
        step++;

        if(step % 100 == 0)
        {
            std::cout << "Current time: " << totalTime << ", End time: " << endTime << ", Step: " << step << std::endl;
        }
    }

    std::cout << "Copying results back." << std::endl;
    // Copy final data back to host for verification
    alpaka::memcpy(queue, hDensity, dDensity);
    alpaka::memcpy(queue, hEnergy, dEnergy);
    alpaka::memcpy(queue, hPressure, dPressure);
    alpaka::memcpy(queue, hVelocityX, dVelocityX);
    alpaka::memcpy(queue, hVelocityY, dVelocityY);
    alpaka::memcpy(queue, hVelocityZ, dVelocityZ);
    alpaka::memcpy(queue, hFluxDensity, dFluxDensity);
    alpaka::memcpy(queue, hFluxEnergy, dFluxEnergy);
    alpaka::memcpy(queue, hFluxVelocityX, dFluxVelocityX);
    alpaka::memcpy(queue, hFluxVelocityY, dFluxVelocityY);
    alpaka::memcpy(queue, hFluxVelocityZ, dFluxVelocityZ);
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
    auto mdHostVelocityZ = alpaka::experimental::getMdSpan(hVelocityZ);

    bool success = true;
    for(Idx i = 0; i < nx; ++i)
    {
        for(Idx j = 0; j < ny; ++j)
        {
            for(Idx k = 0; k < nz; ++k)
            {
                if(std::isnan(mdHostDensity(i, j, k)) || std::isinf(mdHostDensity(i, j, k))
                   || std::isnan(mdHostEnergy(i, j, k)) || std::isinf(mdHostEnergy(i, j, k))
                   || std::isnan(mdHostPressure(i, j, k)) || std::isinf(mdHostPressure(i, j, k))
                   || std::isnan(mdHostVelocityX(i, j, k)) || std::isinf(mdHostVelocityX(i, j, k))
                   || std::isnan(mdHostVelocityY(i, j, k)) || std::isinf(mdHostVelocityY(i, j, k))
                   || std::isnan(mdHostVelocityZ(i, j, k)) || std::isinf(mdHostVelocityZ(i, j, k)))
                {
                    success = false;
                    break;
                }
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

using TestAccs3D = alpaka::test::EnabledAccs<alpaka::DimInt<3u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("TEST: Cloverleaf", "[benchmark-test]", TestAccs3D)
{
    using Acc = TestType;
    if constexpr(alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt>)
    {
        // Run tests for float data type
        testKernels<Acc, float>();
    }
}
