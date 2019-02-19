/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>

#include <future>
#include <thread>


//-----------------------------------------------------------------------------
struct TestTemplateEmpty
{
template< typename TDevQueue >
void operator()()
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    CHECK(alpaka::queue::empty(f.m_queue));
}
};

//-----------------------------------------------------------------------------
struct TestTemplateCallback
{
template< typename TDevQueue >
void operator()()
{
// Workaround: Clang can not support this when natively compiling device code. See ConcurrentExecPool.hpp.
#if !(BOOST_COMP_CLANG_CUDA && BOOST_ARCH_PTX)
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    std::promise<bool> promise;

    alpaka::queue::enqueue(
        f.m_queue,
        [&](){
            promise.set_value(true);
        }
    );

    CHECK(promise.get_future().get());
#endif
}
};

//-----------------------------------------------------------------------------
struct TestTemplateWait
{
template< typename TDevQueue >
void operator()()
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    bool CallbackFinished = false;
    alpaka::queue::enqueue(
        f.m_queue,
        [&CallbackFinished]() noexcept
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            CallbackFinished = true;
        });

    alpaka::wait::wait(f.m_queue);
    CHECK(CallbackFinished);
}
};

//-----------------------------------------------------------------------------
struct TestTemplateExecNotEmpty
{
template< typename TDevQueue >
void operator()()
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    bool CallbackFinished = false;
    alpaka::queue::enqueue(
        f.m_queue,
        [&f, &CallbackFinished]() noexcept
        {
            CHECK(!alpaka::queue::empty(f.m_queue));
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            CallbackFinished = true;
        });

    // A synchronous queue will always stay empty because the task has been executed immediately.
    if(!alpaka::test::queue::IsSyncQueue<typename Fixture::Queue>::value)
    {
        alpaka::wait::wait(f.m_queue);
    }

    CHECK(alpaka::queue::empty(f.m_queue));
    CHECK(CallbackFinished);
}
};

using TestQueues = alpaka::test::queue::TestQueues;

TEST_CASE( "queueIsInitiallyEmpty", "[queue]")
{
    alpaka::meta::forEachType< TestQueues >( TestTemplateEmpty() );
}

TEST_CASE( "queueCallbackIsWorking", "[queue]")
{
    alpaka::meta::forEachType< TestQueues >( TestTemplateCallback() );
}

TEST_CASE( "queueWaitShouldWork", "[queue]")
{
    alpaka::meta::forEachType< TestQueues >( TestTemplateWait() );
}

TEST_CASE( "queueShouldNotBeEmptyWhenLastTaskIsStillExecutingAndIsEmptyAfterProcessingFinished", "[queue]")
{
    alpaka::meta::forEachType< TestQueues >( TestTemplateExecNotEmpty() );
}
