/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/core/Unused.hpp>

#include <catch2/catch.hpp>

TEST_CASE("ompScheduleDefaultConstructor", "[core]")
{
    auto const schedule = alpaka::omp::Schedule{};
    alpaka::ignore_unused(schedule);
}

TEST_CASE("ompScheduleConstructor", "[core]")
{
    auto const static_schedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Static, 5};
    alpaka::ignore_unused(static_schedule);

    auto const guided_schedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Guided};
    alpaka::ignore_unused(guided_schedule);
}

TEST_CASE("ompScheduleConstexprConstructor", "[core]")
{
    constexpr auto schedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Dynamic};
    alpaka::ignore_unused(schedule);
}

TEST_CASE("ompGetSchedule", "[core]")
{
    auto const schedule = alpaka::omp::getSchedule();
    alpaka::ignore_unused(schedule);
}

TEST_CASE("ompSetSchedule", "[core]")
{
    auto const expected_schedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Dynamic, 3};
    alpaka::omp::setSchedule(expected_schedule);
    // The check makes sense only when this feature is supported
#if defined _OPENMP && _OPENMP >= 200805 && defined ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    auto const actual_schedule = alpaka::omp::getSchedule();
    REQUIRE(expected_schedule.kind == actual_schedule.kind);
    REQUIRE(expected_schedule.chunkSize == actual_schedule.chunkSize);
#endif
}

TEST_CASE("ompSetNoSchedule", "[core]")
{
    auto const expected_schedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Guided, 2};
    alpaka::omp::setSchedule(expected_schedule);
    auto const no_schedule = alpaka::omp::Schedule{alpaka::omp::Schedule::NoSchedule};
    alpaka::omp::setSchedule(no_schedule);
    // The check makes sense only when this feature is supported
#if defined _OPENMP && _OPENMP >= 200805 && defined ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    auto const actual_schedule = alpaka::omp::getSchedule();
    REQUIRE(expected_schedule.kind == actual_schedule.kind);
    REQUIRE(expected_schedule.chunkSize == actual_schedule.chunkSize);
#endif
}
