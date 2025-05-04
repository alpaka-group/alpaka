
#pragma once

#include "alpaka/dev/Traits.hpp"

#include <string>
#include <vector>

namespace alpaka
{
    class DeviceProperties
    {
        std::string name;
        std::size_t totalGlobalMem;
        std::vector<std::size_t> warpSizes;
        std::size_t preferredWarpSize;

    public:
        DeviceProperties() = default;

        template<typename TDev, typename TSfinae>
        friend struct trait::GetName;

        template<typename TDev, typename TSfinae>
        friend struct trait::GetMemBytes;

        template<typename TDev, typename TSfinae>
        friend struct trait::GetFreeMemBytes;

        template<typename TDev, typename TSfinae>
        friend struct trait::GetWarpSizes;

        template<typename TDev, typename TSfinae>
        friend struct trait::GetPreferredWarpSize;
    };
} // namespace alpaka
