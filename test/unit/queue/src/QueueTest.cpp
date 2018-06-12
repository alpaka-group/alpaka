/**
 * \file
 * Copyright 2017 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/queue/QueueTestFixture.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <future>
#include <thread>

BOOST_AUTO_TEST_SUITE(queue)


//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueIsInitiallyEmpty,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    BOOST_REQUIRE_EQUAL(true, alpaka::queue::empty(f.m_queue));
}

// gcc 5.4 in combination with nvcc 8.0 fails to compile those tests when --expt-relaxed-constexpr is enabled
// /usr/include/c++/5/tuple(484) (col. 17): error: calling a __host__ function("std::_Tuple_impl<(unsigned long)0ul,  :: > ::_Tuple_impl [subobject]") from a __device__ function("std::tuple< :: > ::tuple") is not allowed
#if !((BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(5, 0, 0)) && (BOOST_COMP_NVCC == BOOST_VERSION_NUMBER(8, 0, 0)))
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueCallbackIsWorking,
    TDevQueue,
    alpaka::test::queue::TestQueues)
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

    BOOST_REQUIRE_EQUAL(true, promise.get_future().get());
#endif
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueShouldBeEmptyAfterProcessingFinished,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    bool CallbackFinished = false;

    alpaka::queue::enqueue(f.m_queue, [&CallbackFinished]() noexcept {CallbackFinished = true;});

    // A synchronous queue will always stay empty because the task has been executed immediately.
    if(!alpaka::test::queue::IsSyncQueue<typename Fixture::Queue>::value)
    {
        // Wait that the queue finally gets empty again.
        while(!alpaka::queue::empty(f.m_queue))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    BOOST_REQUIRE_EQUAL(true, alpaka::queue::empty(f.m_queue));
    BOOST_REQUIRE_EQUAL(true, CallbackFinished);
}
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    queueWaitShouldWork,
    TDevQueue,
    alpaka::test::queue::TestQueues)
{
    using Fixture = alpaka::test::queue::QueueTestFixture<TDevQueue>;
    Fixture f;

    // TODO: better add some long running (~0.5s) task here

    bool CallbackFinished = false;
    alpaka::queue::enqueue(f.m_queue, [&CallbackFinished]() noexcept {CallbackFinished = true;});

    alpaka::wait::wait(f.m_queue);
    BOOST_REQUIRE_EQUAL(true, CallbackFinished);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
