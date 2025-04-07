
#pragma once

#include "alpaka/dev/Traits.hpp"

#include <optional>
#include <string>
#include <vector>

namespace alpaka
{
    class DeviceProperties
    {
        std::optional<std::string> name;
        std::optional<std::size_t> totalGlobalMem;
        std::optional<std::size_t> freeGlobalMem;
        std::optional<std::vector<std::size_t>> warpSizes;
        std::optional<std::size_t> preferredWarpSize;

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
