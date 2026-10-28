/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#if ALPAKA_OS_WINDOWS
#    include <direct.h>
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

using namespace alpaka;

// Declare the function type we expect
using AlpakaMainT = int(int argc, char* argv[]);

std::string toLower(std::string str)
{
    std::ranges::transform(str, str.begin(), [](unsigned char c) { return std::tolower(c); });
    return str;
}

int example(int argc, char* argv[], auto const usedApi)
{
    std::filesystem::path p = argv[0];
    auto binaryPath = p.parent_path().string();

#if ALPAKA_OS_WINDOWS
    HINSTANCE handle;
#else
    void* handle = nullptr;
#endif

    // try to load first in working directory, then in binary directory, then via LD_LIBRARY_PATH
    auto prefixList = std::vector<std::string>{"./", binaryPath + "/", ""};
    for(auto libPrefix : prefixList)
    {
        auto libraryName = libPrefix + "libalpaka-ls_" + onHost::getName(usedApi);

#if ALPAKA_OS_WINDOWS
        libraryName += ".dll";
        handle = LoadLibrary(TEXT((libraryName).c_str()));
#else
#    if ALPAKA_OS_IOS
        libraryName += ".dylib";
#    else
        libraryName += ".so";
#    endif

        handle = dlopen(toLower(libraryName).c_str(), RTLD_NOW | RTLD_LAZY | RTLD_GLOBAL);
#endif
        if(handle)
            break;
    }
    if(handle)
    {
        // Lookup function
#if ALPAKA_OS_WINDOWS
        fn_startBackend = (startBackend_t) GetProcAddress(libBackend, "xmrstak_start_backend");
        if(!func)
        {
            std::cerr << "dlsym failed: " << GetLastError() << "\n";
            FreeLibrary(handle);
        }
#else
        auto func = reinterpret_cast<AlpakaMainT*>(
            dlsym(handle, "alpakaMain") // <-- if compiled without extern "C", use mangled
        );
        if(!func)
        {
            std::cerr << "dlsym failed: " << dlerror() << "\n";
            dlclose(handle);
        }
#endif

        if(func)
        {
            // Forward argc/argv from main to alpakaMain
            [[maybe_unused]] int ret = func(argc, argv);
#if ALPAKA_OS_WINDOWS
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
        }
    }

    return EXIT_SUCCESS;
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [-h]" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    int opt;
    while((opt = getopt(argc, argv, "h")) != -1)
    {
        switch(opt)
        {
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return onHost::executeForEach([=](auto const& api) { return example(argc, argv, api); }, onHost::apis);
}
