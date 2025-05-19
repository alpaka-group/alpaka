
#pragma once

#include "alpaka/dev/Traits.hpp"

#include <string>
#include <vector>

namespace alpaka
{

    struct DeviceProperties
    {
        std::string name;
        std::size_t totalGlobalMem;
        std::vector<std::size_t> warpSizes;
        std::size_t preferredWarpSize;
    };
} // namespace alpaka
