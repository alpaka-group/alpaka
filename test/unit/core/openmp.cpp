/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <tuple>
#include <type_traits>

using namespace alpaka;

TEST_CASE("OpenMP capability matches CMake configuration", "[core][openmp]")
{
    CHECK(static_cast<bool>(ALPAKA_OMP) == static_cast<bool>(ALPAKA_TEST_EXPECT_OMP));

    constexpr auto backends = onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors);
    bool hasCpuOmpBlocks = false;
    std::apply(
        [&](auto... backend)
        {
            alpaka::unused(backend...);
            ((hasCpuOmpBlocks |= std::is_same_v<ALPAKA_TYPEOF(getApi(backend)), api::Host>
                                 && std::is_same_v<ALPAKA_TYPEOF(getDeviceKind(backend)), deviceKind::Cpu>
                                 && std::is_same_v<ALPAKA_TYPEOF(getExecutor(backend)), exec::CpuOmpBlocks>),
             ...);
        },
        backends);
    CHECK(hasCpuOmpBlocks == static_cast<bool>(ALPAKA_TEST_EXPECT_CPU_OMP));
}
