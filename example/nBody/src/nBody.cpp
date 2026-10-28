/* Copyright 2025 Anton Reinhard, Tim Hanel
 * SPDX-License-Identifier: ISC
 */

#include "Kernels.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "common.hpp"
#include "helpers.hpp"
#include "particles.hpp"

#ifdef PNGWRITER_ENABLED
#    include "writepng.hpp"
#endif

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace alpaka::example::nBody
{
    // Appropriate chunk size to split your problem for your Acc
    constexpr auto chunkSize = CVec<IdxType, 256_idx>{};

    /** Parameters of a simulation run. */
    struct SimulationParams
    {
        //! The number of particles to simulate.
        IdxType numParticles;
        //! The number of time steps to run for.
        IdxType numTimeSteps;
        //! The delta t to use as time steps.
        BaseType dt;
    };

    /** Parameters used if they are not overwritten by command line arguments. */
    static SimulationParams defaultSimulationParams()
    {
        return SimulationParams{
            .numParticles = defaultNumParticles,
            .numTimeSteps = defaultTimeSteps,
            .dt = defaultDt};
    }

    void printExampleHeader(bool const writePngs, bool const benchmarkMode, SimulationParams const& params)
    {
        if(!benchmarkMode)
        {
            std::cout << "================================" << '\n';
            std::cout << "Example n-Body simulation" << '\n';
            std::cout << "    Number of particles [#]: " << params.numParticles << '\n';
            std::cout << "    Time steps [#]: " << params.numTimeSteps << '\n';
            std::cout << "    dt: " << params.dt << '\n';
            if(writePngs)
                std::cout << "    Writing pngs to disk" << '\n';
            else
                std::cout << "    Not writing pngs to disk" << '\n';
            std::cout << "================================" << '\n';
            std::cout << '\n';
        }
        else
        {
            // print csv header
            std::cout << "# N-Body simulation, " << timeStepsBenchmark << " timesteps per run" << '\n';
            std::cout << "accelerator,numParticles,timePerStep(s),GFLOPs" << '\n';
        }
    }

    /** @brief Run an n-body simulation.
     *
     * Some number of particles are simulated in 3-dimensional space. Each particle gets a mass, position vector, and a
     * velocity vector, initialized randomly (see common.hpp for constants to tweak the generation). The particles then
     * interact gravitationally. The UpdateVelocitiesKernel updates each particle's velocity for one timestep and
     * writes them back. The updated velocities only depend on the other particles' positions and masses, so there is
     * no data race. Next, the UpdatePositionsKernel moves each particle independently according to its current
     * velocity. This repeats for the desired number of time steps.
     *
     * @param deviceSpec The device specification to run on.
     * @param computeExec The device to execute on.
     * @param writePngs Whether to write pngs to disk or not.
     * @param benchmarkMode Whether to run in benchmark mode. See the help docs below.
     * @param params The number of particles, time steps and delta t of the simulation.
     */
    int example(
        auto const deviceSpec,
        auto const computeExec,
        bool const writePngs,
        bool const benchmarkMode,
        SimulationParams const& params)
    {
        using namespace alpaka;
        using namespace alpaka::onHost;

        using IdxTypeVec = Vec<IdxType, 1_idx>;

        if(!benchmarkMode)
            std::cout << "\nRunning accelerator: " << deviceSpec.getApi().getName() << std::endl;

        auto devSelector = makeDeviceSelector(deviceSpec);
        Device devAcc = devSelector.makeDevice(0);

        // simulation defines
        // {Y, X}
        IdxTypeVec const extents = Vec{params.numParticles};

        auto massesHost = onHost::allocHost<BaseType>(extents);
        auto xPositionsHost = onHost::allocHost<BaseType>(extents);
        auto yPositionsHost = onHost::allocHost<BaseType>(extents);
        auto zPositionsHost = onHost::allocHost<BaseType>(extents);
        auto xVelocitiesHost = onHost::allocHost<BaseType>(extents);
        auto yVelocitiesHost = onHost::allocHost<BaseType>(extents);
        auto zVelocitiesHost = onHost::allocHost<BaseType>(extents);

        [[maybe_unused]] auto particleDataHost = ParticleData{
            massesHost.getView(),
            xPositionsHost.getView(),
            yPositionsHost.getView(),
            zPositionsHost.getView(),
            xVelocitiesHost.getView(),
            yVelocitiesHost.getView(),
            zVelocitiesHost.getView(),
        };

        if(!benchmarkMode)
            std::cout << "Initializing data on host" << '\n';
        initMasses(massesHost);
        initPositions(xPositionsHost, yPositionsHost, zPositionsHost);
        initVelocities(
            xVelocitiesHost,
            yVelocitiesHost,
            zVelocitiesHost,
            xPositionsHost,
            yPositionsHost,
            zPositionsHost);

#ifdef PNGWRITER_ENABLED
        auto colors = onHost::allocHost<Color>(extents);
        if(writePngs)
            initColors(colors);
#else
        if(writePngs)
            std::cout << "WritePngs is enabled, but the pngwriter library was not found. Not writing pngs to disk."
                      << '\n';
#endif

        if(!benchmarkMode)
            std::cout << "Done initializing data on host" << '\n';

        auto massesDev = onHost::allocLike(devAcc, massesHost);
        auto xPositionsDev = onHost::allocLike(devAcc, xPositionsHost);
        auto yPositionsDev = onHost::allocLike(devAcc, yPositionsHost);
        auto zPositionsDev = onHost::allocLike(devAcc, zPositionsHost);
        auto xVelocitiesDev = onHost::allocLike(devAcc, xVelocitiesHost);
        auto yVelocitiesDev = onHost::allocLike(devAcc, yVelocitiesHost);
        auto zVelocitiesDev = onHost::allocLike(devAcc, zVelocitiesHost);

        // Select queue
        Queue dumpQueue = devAcc.makeQueue();
        Queue computeQueue = devAcc.makeQueue();

        // copy data to device
        onHost::memcpy(dumpQueue, massesDev, massesHost);
        onHost::memcpy(dumpQueue, xPositionsDev, xPositionsHost);
        onHost::memcpy(dumpQueue, yPositionsDev, yPositionsHost);
        onHost::memcpy(dumpQueue, zPositionsDev, zPositionsHost);
        onHost::memcpy(dumpQueue, xVelocitiesDev, xVelocitiesHost);
        onHost::memcpy(dumpQueue, yVelocitiesDev, yVelocitiesHost);
        onHost::memcpy(dumpQueue, zVelocitiesDev, zVelocitiesHost);
        onHost::wait(dumpQueue);
        if(!benchmarkMode)
            std::cout << "Data copied to device" << '\n';

        auto particleData = ParticleData{
            massesDev.getView(),
            xPositionsDev.getView(),
            yPositionsDev.getView(),
            zPositionsDev.getView(),
            xVelocitiesDev.getView(),
            yVelocitiesDev.getView(),
            zVelocitiesDev.getView()};

        auto const numChunks = divCeil(Vec{extents.x()}, chunkSize);

        // The frame spec describes the logical parallelism exposed to the kernel, not the exact mapping on the threads
        // and the blocks of the backend.
        auto const frameSpec = FrameSpec{numChunks, chunkSize, computeExec};

        // Make an instance of the kernel object to use later
        UpdateVelocitiesKernel updateVelocitiesKernel;
        UpdatePositions updatePositionsKernel{params.dt};

        auto const startTime = std::chrono::high_resolution_clock::now();

        // Simulate
        for(IdxType step = 1; step <= params.numTimeSteps; ++step)
        {
            // Queue one step of the simulation
            // The kernel bundle contains the kernel first, then all its arguments *except* the accelerator, which is
            // implicitly passed by alpaka.
            computeQueue.enqueue(frameSpec, KernelBundle{updateVelocitiesKernel, particleData, chunkSize, params.dt});

            // execute velocity updates using simd concurrency
            onHost::concurrent<BaseType>(
                computeQueue,
                particleData.xPositions.getExtents(),
                updatePositionsKernel,
                particleData.xPositions,
                particleData.yPositions,
                particleData.zPositions,
                particleData.xVelocities,
                particleData.yVelocities,
                particleData.zVelocities);

#ifdef PNGWRITER_ENABLED
            if(step % pngStepSize == 0 && writePngs)
            {
                wait(computeQueue);

                // copy data to host
                onHost::memcpy(dumpQueue, massesHost, particleData.masses);
                onHost::memcpy(dumpQueue, xPositionsHost, particleData.xPositions);
                onHost::memcpy(dumpQueue, yPositionsHost, particleData.yPositions);
                onHost::memcpy(dumpQueue, zPositionsHost, particleData.zPositions);
                onHost::memcpy(dumpQueue, xVelocitiesHost, particleData.xVelocities);
                onHost::memcpy(dumpQueue, yVelocitiesHost, particleData.yVelocities);
                onHost::memcpy(dumpQueue, zVelocitiesHost, particleData.zVelocities);
                onHost::wait(dumpQueue);

                wait(dumpQueue);

                std::ostringstream oss;
                oss << std::setw(5) << std::setfill('0') << (step / pngStepSize);

                writePng(particleDataHost, colors, "particles_" + oss.str() + ".png");
            }
#endif
        }

        // Wait for the entire queue to be finished
        wait(computeQueue);
        auto const endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - startTime;

        auto timePerStep = elapsedTime.count() / params.numTimeSteps;
        if(!benchmarkMode)
        {
            std::cout << "The simulation took " << elapsedTime.count() << " seconds." << '\n';
            std::cout << "Time per time step: " << timePerStep * 1000 << " ms." << '\n';
            std::cout << "GFLOP needed per time step: " << flopsRequiredPerTimeStep(params.numParticles) << '\n';
            std::cout << "Average achieved GFLOPS: "
                      << static_cast<double>(flopsRequiredPerTimeStep(params.numParticles)) / timePerStep / 1e9
                      << '\n';
        }
        else
        {
            std::cout << deviceSpec.getApi().getName() << "," << params.numParticles << "," << timePerStep << ","
                      << static_cast<double>(flopsRequiredPerTimeStep(params.numParticles)) / timePerStep / 1e9
                      << std::endl;
        }

        return EXIT_SUCCESS;
    }

    int benchmark(auto const deviceSpec, auto const computeExec, BaseType const dt)
    {
        for(auto numParticles : numParticlesBenchmark)
        {
            SimulationParams params;
            params.numParticles = numParticles;
            params.numTimeSteps = timeStepsBenchmark;
            params.dt = dt;
            example(deviceSpec, computeExec, false, true, params);
        }
        return EXIT_SUCCESS;
    }

    void help(char* argv[])
    {
        std::cerr << argv[0] << " [OPTIONS]" << '\n';
        std::cerr << "  -n numParticles: Number of particles. Default: " << defaultNumParticles << '\n';
        std::cerr << "  -t numTimeSteps: Number of time steps that the simulation is run for. Default: "
                  << defaultTimeSteps << '\n';
        std::cerr << "  -d dt: Delta t for the timesteps. Default: " << defaultDt << '\n';
        std::cerr
            << "  -p: Write pngs of the particles to disk. Has no effect when pngwriter is not installed. Default: off"
            << '\n';
        std::cerr << "  -b: Benchmark mode. In this mode, csv formatted results are printed to stdout and multiple "
                     "predefined problem sizes are run. Other options are ignored in this mode. Default: off"
                  << '\n';
        std::cerr << "  -h: Print this help message" << '\n';
        std::cerr << '\n';
    }

} // namespace alpaka::example::nBody

auto main(int argc, char* argv[]) -> int
{
    using namespace alpaka;
    using namespace alpaka::example::nBody;

    // Default value if no command line argument used
    SimulationParams params = defaultSimulationParams();
    bool writePngs = false;
    bool benchmarkMode = false;

    int opt;

    while((opt = getopt(argc, argv, "hn:t:d:pb")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                params.numParticles = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));

                auto adjustedParticles = divCeil(params.numParticles, chunkSize.x()) * chunkSize.x();
                if(adjustedParticles != params.numParticles)
                {
                    std::cout << "Adjusting number of particles to " << adjustedParticles
                              << " to be a multiple of the chunk size (" << chunkSize.x() << ")" << '\n';
                    params.numParticles = adjustedParticles;
                }

                if(params.numParticles < 1)
                {
                    throw std::invalid_argument("Number of particles must be at least 1");
                }
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 't':
            try
            {
                params.numTimeSteps = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'd':
            try
            {
                params.dt = static_cast<BaseType>(std::stod(optarg, nullptr));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'p':
            writePngs = true;
            break;
        case 'b':
            benchmarkMode = true;
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    printExampleHeader(writePngs, benchmarkMode, params);

    if(benchmarkMode)
    {
        return onHost::executeForEach(
            [=](alpaka::concepts::BackendSpec auto const& backend)
            {
                auto selector = onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend});
                if(!selector.isAvailable())
                    return EXIT_SUCCESS;
                return alpaka::example::nBody::benchmark(
                    alpaka::onHost::DeviceSpec{backend},
                    alpaka::getExecutor(backend),
                    params.dt);
            },
            onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
    }
    else
    {
        return onHost::executeForEach(
            [=](alpaka::concepts::BackendSpec auto const& backend)
            {
                auto selector = onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend});
                if(!selector.isAvailable())
                    return EXIT_SUCCESS;
                return alpaka::example::nBody::example(
                    alpaka::onHost::DeviceSpec{backend},
                    alpaka::getExecutor(backend),
                    writePngs,
                    benchmarkMode,
                    params);
            },
            onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
    }
}
