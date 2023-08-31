/* Copyright 2023 Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/ThreadPool.hpp>

#include <catch2/catch_test_macros.hpp>

#if defined(__has_feature)
#    if __has_feature(thread_sanitizer)
#        define TSAN
#    endif
#endif

TEST_CASE("threadpool", "[core]")
{
    alpaka::core::detail::ThreadPool tp{2};

    auto f1 = tp.enqueueTask([] { throw std::runtime_error("42"); });
    auto f2 = tp.enqueueTask([] { throw 42; });
    auto f3 = tp.enqueueTask([]() noexcept {});

    CHECK_THROWS_AS(f1.get(), std::runtime_error);

#ifdef TSAN
#    warning "Part of the threadpool test fails the TSAN CI and is therefore disabled."
    // TODO(bgruber): Revisit this in the future on a new CI image. This problem does not happen locally.
#else
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
