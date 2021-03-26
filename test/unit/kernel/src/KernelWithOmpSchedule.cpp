/* Copyright 2020-2021 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <cstdint>

//#############################################################################
// Base kernel, not to be used directly in unit tests
struct KernelWithOmpScheduleBase
{
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // No run-time check is performed
        alpaka::ignore_unused(acc);
        ALPAKA_CHECK(*success, true);
    }
};

// Kernel that sets the schedule kind via constexpr ompScheduleKind.
// Checks that this variable is only declared and not defined, It also tests that alpaka never odr-uses it.
struct KernelWithConstexprMemberOmpScheduleKind : KernelWithOmpScheduleBase
{
    static constexpr auto ompScheduleKind = alpaka::omp::Schedule::Runtime;
};

// Kernel that sets the schedule kind via non-constexpr ompScheduleKind.
struct KernelWithMemberOmpScheduleKind : KernelWithOmpScheduleBase
{
    static const alpaka::omp::Schedule::Kind ompScheduleKind;
};
// In this case, the member has to be defined externally
const alpaka::omp::Schedule::Kind KernelWithMemberOmpScheduleKind::ompScheduleKind = alpaka::omp::Schedule::NoSchedule;

// Kernel that sets the schedule chunk size via constexpr ompScheduleChunkSize.
// Checks that this variable is only declared and not defined, It also tests that alpaka never odr-uses it.
struct KernelWithConstexprStaticMemberOmpScheduleChunkSize : KernelWithOmpScheduleBase
{
    static constexpr int ompScheduleChiunkSize = 5;
};

// Kernel that sets the schedule chunk size via non-constexpr ompScheduleChunkSize.
struct KernelWithStaticMemberOmpScheduleChunkSize : KernelWithOmpScheduleBase
{
    static const int ompScheduleChunkSize;
};
// In this case, the member has to be defined externally
const int KernelWithStaticMemberOmpScheduleChunkSize::ompScheduleChunkSize = 2;

// Kernel that sets the schedule chunk size via non-constexpr non-static ompScheduleChunkSize.
struct KernelWithMemberOmpScheduleChunkSize : KernelWithOmpScheduleBase
{
    int ompScheduleChunkSize = 4;
};

// Kernel that sets the schedule via partial specialization of a trait
struct KernelWithTraitOmpSchedule : KernelWithOmpScheduleBase
{
};

// Kernel that sets the schedule via both member and partial specialization of a trait.
// In this case test that the trait is used, not the member.
struct KernelWithMemberAndTraitOmpSchedule : KernelWithOmpScheduleBase
{
    static constexpr auto ompScheduleKind = alpaka::omp::Schedule::Dynamic;
};

namespace alpaka
{
    namespace traits
    {
        // Specialize the trait for all kernels
        template<typename TKernelFnObj, typename TAcc>
        struct OmpSchedule<TKernelFnObj, TAcc>
        {
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST static auto getOmpSchedule(
                TKernelFnObj const& kernelFnObj,
                Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
                Vec<TDim, Idx<TAcc>> const& threadElemExtent,
                TArgs const&... args) -> alpaka::omp::Schedule
            {
                alpaka::ignore_unused(kernelFnObj);
                alpaka::ignore_unused(blockThreadExtent);
                alpaka::ignore_unused(threadElemExtent);
                alpaka::ignore_unused(args...);

                return alpaka::omp::Schedule{alpaka::omp::Schedule::Guided, 2};
            }
        };
    } // namespace traits
} // namespace alpaka

// Generic testing routine for the given kernel type
template<typename TAcc, typename TKernel>
void test()
{
    using Acc = TAcc;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    TKernel kernel;

    REQUIRE(fixture(kernel));
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstexprMemberOmpScheduleKind", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithConstexprMemberOmpScheduleKind>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithMemberOmpScheduleKind", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithMemberOmpScheduleKind>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstexprStaticMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithConstexprStaticMemberOmpScheduleChunkSize>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithStaticMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithStaticMemberOmpScheduleChunkSize>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithMemberOmpScheduleChunkSize>();
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("kernelWithTraitOmpSchedule", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithTraitOmpSchedule>();
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("kernelWithMemberAndTraitOmpSchedule", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithMemberAndTraitOmpSchedule>();
}
