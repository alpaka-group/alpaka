/* Copyright 2020-2021 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <cstdint>

// Base kernel, not to be used directly in unit tests
struct KernelWithOmpScheduleBase
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // No run-time check is performed
        alpaka::ignore_unused(acc);
        ALPAKA_CHECK(*success, true);
    }
};

// Kernel that sets the schedule kind via constexpr ompScheduleKind, but no chunk size.
// Checks that this variable is only declared and not defined, It also tests that alpaka never odr-uses it.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithConstexprMemberOmpScheduleKind : KernelWithOmpScheduleBase
{
    static constexpr auto ompScheduleKind = TKind;
};

// Kernel that sets the schedule kind via non-constexpr ompScheduleKind, but no chunk size.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithMemberOmpScheduleKind : KernelWithOmpScheduleBase
{
    static const alpaka::omp::Schedule::Kind ompScheduleKind = TKind;
};

// Kernel that sets the schedule chunk size via constexpr ompScheduleChunkSize in addition to schedule kind, but no
// chunk size. Checks that this variable is only declared and not defined, It also tests that alpaka never odr-uses it.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithConstexprStaticMemberOmpScheduleChunkSize : KernelWithConstexprMemberOmpScheduleKind<TKind>
{
    static constexpr int omp_schedule_chunk_size = 5;
};

// Kernel that sets the schedule chunk size via non-constexpr ompScheduleChunkSize in addition to schedule kind.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithStaticMemberOmpScheduleChunkSize : KernelWithMemberOmpScheduleKind<TKind>
{
    static const int omp_schedule_chunk_size;
};
// In this case, the member has to be defined externally
template<alpaka::omp::Schedule::Kind TKind>
const int KernelWithStaticMemberOmpScheduleChunkSize<TKind>::omp_schedule_chunk_size = 2;

// Kernel that sets the schedule chunk size via non-constexpr non-static ompScheduleChunkSize in addition to schedule
// kind.
template<alpaka::omp::Schedule::Kind TKind>
struct KernelWithMemberOmpScheduleChunkSize : KernelWithConstexprMemberOmpScheduleKind<TKind>
{
    int m_omp_schedule_chunk_size = 4;
};

// Kernel that copies the given base kernel and adds an OmpSchedule trait on top
template<typename TBase>
struct KernelWithTrait : TBase
{
};

namespace alpaka
{
    namespace traits
    {
        // Specialize the trait for kernels of type KernelWithTrait<>
        template<typename TBase, typename TAcc>
        struct OmpSchedule<KernelWithTrait<TBase>, TAcc>
        {
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST static auto get_omp_schedule(
                KernelWithTrait<TBase> const& kernel_fn_obj,
                Vec<TDim, Idx<TAcc>> const& block_thread_extent,
                Vec<TDim, Idx<TAcc>> const& thread_elem_extent,
                TArgs const&... args) -> alpaka::omp::Schedule
            {
                alpaka::ignore_unused(kernel_fn_obj);
                alpaka::ignore_unused(block_thread_extent);
                alpaka::ignore_unused(thread_elem_extent);
                alpaka::ignore_unused(args...);

                return alpaka::omp::Schedule{alpaka::omp::Schedule::Static, 4};
            }
        };
    } // namespace traits
} // namespace alpaka

// Generic testing routine for the given kernel type
template<typename TAcc, typename TKernel>
void test_kernel()
{
    using Acc = TAcc;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    // Base version with no OmpSchedule trait
    TKernel kernel;
    REQUIRE(fixture(kernel));

    // Same members, but with OmpSchedule trait
    KernelWithTrait<TKernel> kernel_with_trait;
    REQUIRE(fixture(kernel_with_trait));
}

// Note: it turned out not possible to test all possible combinations as it causes several compilers to crash in CI.
// However the following tests should cover all important cases

TEMPLATE_LIST_TEST_CASE("kernelWithOmpScheduleBase", "[kernel]", alpaka::test::TestAccs)
{
    test_kernel<TestType, KernelWithOmpScheduleBase>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstexprMemberOmpScheduleKind", "[kernel]", alpaka::test::TestAccs)
{
    test_kernel<TestType, KernelWithConstexprMemberOmpScheduleKind<alpaka::omp::Schedule::NoSchedule>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithMemberOmpScheduleKind", "[kernel]", alpaka::test::TestAccs)
{
    test_kernel<TestType, KernelWithMemberOmpScheduleKind<alpaka::omp::Schedule::Static>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithConstexprStaticMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
    test_kernel<TestType, KernelWithConstexprStaticMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Dynamic>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithStaticMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
    test_kernel<TestType, KernelWithStaticMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Guided>>();
}

TEMPLATE_LIST_TEST_CASE("kernelWithMemberOmpScheduleChunkSize", "[kernel]", alpaka::test::TestAccs)
{
#if defined _OPENMP && _OPENMP >= 200805
    test_kernel<TestType, KernelWithMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Auto>>();
#endif
    test_kernel<TestType, KernelWithMemberOmpScheduleChunkSize<alpaka::omp::Schedule::Runtime>>();
}
