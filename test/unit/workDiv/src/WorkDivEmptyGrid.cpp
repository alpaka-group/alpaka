/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#include <alpaka/acc/AccCpuOmp2Threads.hpp>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuTbbBlocks.hpp>
#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/kernel/KernelBundle.hpp>
#include <alpaka/kernel/KernelFunctionAttributes.hpp>
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

struct TestKernelAbort
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const&) const
    {
        ALPAKA_ASSERT_ACC(false);
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("emptyWorkDiv.1D", "[emptyWorkDiv]", TestAccs)
{
    using Acc = TestType;
    using Device = alpaka::Dev<Acc>;
    using Platform = alpaka::Platform<Device>;
    using Queue = alpaka::Queue<Device, alpaka::Blocking>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    const Platform platform{};
    const Device dev = alpaka::getDevByIdx(platform, 0);
    Queue queue = alpaka::Queue<Device, alpaka::Blocking>{dev};

    WorkDiv empty_grid{0u, 1u, 1u};
    CHECK_NOTHROW(alpaka::exec<Acc>(queue, empty_grid, TestKernelAbort{}));

    CHECK_NOTHROW(alpaka::wait(queue));
}

using TestAccs3D = alpaka::test::EnabledAccs<alpaka::DimInt<3u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("emptyWorkDiv.3D", "[emptyWorkDiv]", TestAccs3D)
{
    using Acc = TestType;
    using Device = alpaka::Dev<Acc>;
    using Platform = alpaka::Platform<Device>;
    using Queue = alpaka::Queue<Device, alpaka::Blocking>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    const Platform platform{};
    const Device dev = alpaka::getDevByIdx(platform, 0);
    Queue queue = alpaka::Queue<Device, alpaka::Blocking>{dev};

    WorkDiv empty_grid{{0u, 0u, 0u}, {1u, 1u, 1u}, {1u, 1u, 1u}};
    CHECK_NOTHROW(alpaka::exec<Acc>(queue, empty_grid, TestKernelAbort{}));

    CHECK_NOTHROW(alpaka::wait(queue));
}
