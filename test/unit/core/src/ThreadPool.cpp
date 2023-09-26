/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/ThreadPool.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("threadpool", "[core]")
{
    alpaka::core::detail::ThreadPool tp{2};

    auto f1 = tp.enqueueTask([] { throw std::runtime_error("42"); });
    auto f2 = tp.enqueueTask([] { throw 42; });
    auto f3 = tp.enqueueTask([]() noexcept {});

    CHECK_THROWS_AS(f1.get(), std::runtime_error);

#ifndef ALPAKA_USES_TSAN
    try
    {
        f2.get();
    }
    catch(int i)
    {
        CHECK(i == 42);
    }
#endif

    CHECK_NOTHROW(f3.get());
}
