/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <random>


using namespace alpaka;

// Declare the function type we expect
using AlpakaMainT = int(int argc, char* argv[]);

void forEachDeviceType(auto const deviceSpec)
{
    auto computeDevSelector = onHost::makeDeviceSelector(deviceSpec);
    uint32_t numDevices = computeDevSelector.getDeviceCount();
    std::cout << "api: " << onHost::getName(deviceSpec.getApi()) << "\n";
    std::cout << "\tdevice kind      : " << onHost::getName(deviceSpec.getDeviceKind()) << "\n";
    std::cout << "\tnumber of devices: " << numDevices << "\n";
    for(uint32_t d = 0; d < numDevices; ++d)
    {
        onHost::DeviceProperties devProps = computeDevSelector.getDeviceProperties(d);
        std::cout << "\t\tid                   : " << d << "\n";
        std::cout << "\t\tname                 : " << devProps.getName() << "\n";
        std::cout << "\t\tmulti processor count: " << devProps.multiProcessorCount << "\n";
        std::cout << "\t\twarp size            : " << devProps.warpSize << "\n";
        std::cout << "\t\tmax threads per block: " << devProps.maxThreadsPerBlock << "\n";
        std::cout << "\t\tglobal memory bytes  : " << devProps.globalMemCapacityBytes << "\n";
        auto executors = onHost::getExecutorsList(deviceSpec, exec::enabledExecutors);
        std::cout << "\t\texecutors            : ";
        std::apply(
            [](auto... executors)
            {
                bool first = true;
                ((std::cout << (first ? "" : ", ") << onHost::getName(executors), first = false), ...);
            },
            executors);
        if(std::tuple_size_v<decltype(executors)> == 0)
            std::cout << "none\n";
        else
            std::cout << "\n";
        std::cout << '\n';
    }
    if(numDevices == 0)
        std::cout << '\n';
}

extern "C" int alpakaMain(int argc, char* argv[])
{
    alpaka::unused(argc, argv);
    concepts::Api auto usedApi = api::ALPAKA_USE_API{};


    auto deviceSpecs = onHost::getDeviceSpecsFor(usedApi);
    std::apply([](auto... devSpec) { (forEachDeviceType(devSpec), ...); }, deviceSpecs);

    return EXIT_SUCCESS;
}
