/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */
#include "alpaka/api/host/OmpCollectiveQueue.hpp"
#include "eventHelper.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

/** @file
 *
 * This tests evaluated if events in a queue follows a defined behaviour. Events used to describe dependencies
 * between queues need to guarantee that tasks not start too early.
 * If an event is re-enqueued and used again the event is not allowed to be complete before the last enqueue of the
 * event is executed on the device.
 *
 * @attention: Compared to older alpaka version the tests not using a special implementation of an emulated kernel
 * which can be triggered from the host side. For CUDA in the past a driver method `cuStreamWaitValue32()` was used and
 * for CPU there wars a different implementation. Sycl was not tested at all. This implementation is providing now a
 * unified emulated kernel which is using mapped memory to signal that a kernel which is performing a busy-wait on
 * the device kernel should finish.
 * If the usage of mapped memory is making issues at some point we should switch back to the old implementation.
 *
 * For OneApi and Intel GPUs some tests will not be performed. The GPU Arc770 used for testing provides most likely
 * only a single hardware queue and therefore kernel which has no dependencies are not executed. Most test assumes that
 * queues can run tasks concurrently even if there is already a running kernel.
 */

using namespace alpaka;
using namespace alpaka::test::event;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

static constexpr auto optionalQueueKind =
#if ALPAKA_OMP
    std::tuple{queueKind::ompCollective};
#else
    std::tuple{};
#endif

static constexpr auto testQueueKinds
    = std::tuple_cat(std::tuple{queueKind::blocking, queueKind::nonBlocking}, optionalQueueKind);

struct TestEventPolicy
{
};

template<>
struct alpaka::trait::IsEventPolicy<TestEventPolicy> : std::true_type
{
};

static_assert(alpaka::concepts::EventPolicy<timing::Enabled>);
static_assert(alpaka::concepts::QueuePolicy<timing::Enabled>);
static_assert(alpaka::concepts::Policy<timing::Enabled>);
static_assert(alpaka::concepts::Policy<TestEventPolicy>);

template<typename... T_Policies>
concept CanInstantiatePolicyList = requires { typename PolicyList<T_Policies...>; };

static_assert(CanInstantiatePolicyList<>);
static_assert(!CanInstantiatePolicyList<TestEventPolicy, TestEventPolicy>);

constexpr auto defaultEventPolicies = onHost::EventPolicyList{};
static_assert(alpaka::concepts::EventPolicyList<decltype(defaultEventPolicies)>);
static_assert(defaultEventPolicies.getTiming() == timing::disabled);

constexpr auto testEventPolicies = onHost::EventPolicyList{timing::enabled, TestEventPolicy{}};
static_assert(alpaka::concepts::EventPolicyList<decltype(testEventPolicies)>);
static_assert(!alpaka::concepts::EventPolicyList<TestEventPolicy>);
static_assert(testEventPolicies.getTiming() == timing::enabled);
static_assert(testEventPolicies.hasPolicy(TestEventPolicy{}));
static_assert(!testEventPolicies.hasPolicy(timing::disabled));

template<typename T_Queue, typename T_Event>
concept CanEnqueueEvent = requires(T_Queue const& queue, T_Event const& event) { queue.enqueue(event); };

template<typename T_Queue, typename... T_Policies>
concept CanMakeEvent = requires(T_Queue const& queue, T_Policies... policies) { queue.makeEvent(policies...); };

/** This test takes care that kernel in different queues can run concurrent and if we can communicate between host and
 * the device via mapped memory. Even if the concurrent queue test says true it could be that kernels can run under the
 * condition we test concurrent but if too many blocking kernel are enqueued they can not run concurrent.
 */
TEMPLATE_LIST_TEST_CASE("device analysis", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());

    bool hasCQueue = detectConcurrentQueue(device);
    INFO("Concurrent kernel queue detected: " << (hasCQueue ? "yes" : "no"));
    if(hasCQueue)
        CHECK(hasCQueue);
    else
        CHECK_FALSE(hasCQueue);

    bool supportMappedMemTrigger = mappedMemTriggerDetection(device);
    INFO("Can trigger via mapped memory: " << (supportMappedMemTrigger ? "yes" : "no"));
    if(supportMappedMemTrigger)
        CHECK(supportMappedMemTrigger);
    else
        CHECK_FALSE(supportMappedMemTrigger);
}

TEMPLATE_LIST_TEST_CASE("event creation and enqueue", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());

    onHost::Queue queue = device.makeQueue();
    onHost::Event ev = device.makeEvent(onHost::EventPolicyList{TestEventPolicy{}});
    CHECK(ev.getPolicyList().hasPolicy(TestEventPolicy{}));
    queue.enqueue(ev);
    onHost::wait(ev);
    CHECK(ev.isComplete() == true);
}

struct EmptyTimingKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const&) const
    {
    }
};

TEMPLATE_LIST_TEST_CASE("timing-enabled events measure queued work", "", TestApis)
{
    auto [device, executor] = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto untimedQueue = device.makeQueue();
    auto start = device.makeEvent(timing::enabled);
    auto end = device.makeEvent(timing::enabled);
    auto untimedEvent = device.makeEvent(timing::disabled);
    onHost::concepts::FrameSpec auto const frameSpec = onHost::getFrameSpec(device, executor, Vec{1u});

    static_assert(ALPAKA_TYPEOF(queue.getTiming()){} == timing::enabled);
    static_assert(ALPAKA_TYPEOF(start.getTiming()){} == timing::enabled);
    static_assert(ALPAKA_TYPEOF(untimedEvent.getTiming()){} == timing::disabled);
    static_assert(!CanEnqueueEvent<ALPAKA_TYPEOF(untimedQueue), ALPAKA_TYPEOF(start)>);
    static_assert(!CanMakeEvent<ALPAKA_TYPEOF(queue), timing::Enabled>);
    static_assert(!CanMakeEvent<ALPAKA_TYPEOF(queue), timing::Disabled>);

    queue.enqueue(start);
    queue.enqueue(frameSpec, KernelBundle{EmptyTimingKernel{}});
    queue.enqueue(end);

    auto const elapsed = onHost::getElapsedTime(start, end);
    CHECK(elapsed >= std::chrono::duration<double>::zero());
}

TEMPLATE_LIST_TEST_CASE("elapsed-time query waits for both events", "", TestApis)
{
    // This test makes the end timing event finish before the start event by using two separate queues.
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    auto startQueue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto endQueue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto start = device.makeEvent(timing::enabled);
    auto end = device.makeEvent(timing::enabled);

    std::promise<void> releaseStartPromise;
    auto releaseStart = releaseStartPromise.get_future().share();
    startQueue.enqueueHostFn([releaseStart]() { releaseStart.wait(); });
    startQueue.enqueue(start);
    endQueue.enqueue(end);
    // This ensures that the end event completes before getElapsedTime is called.
    onHost::wait(end);

    std::promise<void> elapsedStartedPromise;
    auto elapsedStarted = elapsedStartedPromise.get_future();
    auto elapsedFuture = std::async(
        std::launch::async,
        [&]()
        {
            elapsedStartedPromise.set_value();
            return onHost::getElapsedTime(start, end);
        });
    elapsedStarted.wait();

    // The start queue is blocked, so getElapsedTime should remain blocked in the asynchronous task.
    auto const elapsedStatus = elapsedFuture.wait_for(std::chrono::milliseconds{100});
    CHECK(elapsedStatus == std::future_status::timeout);

    // Release the start queue task.
    releaseStartPromise.set_value();
    // Since the start event is no longer blocked, getElapsedTime should continue and return the measurement.
    auto const elapsed = elapsedFuture.get();
    CHECK(alpaka::math::abs(elapsed) >= std::chrono::milliseconds{100});
}

TEST_CASE("host elapsed-time query rejects unrecorded events", "")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto recorded = device.makeEvent(timing::enabled);
    auto unrecorded = device.makeEvent(timing::enabled);

    queue.enqueue(recorded);
    onHost::wait(recorded);

    CHECK_THROWS_AS(onHost::getElapsedTime(unrecorded, recorded), std::logic_error);
    CHECK_THROWS_AS(onHost::getElapsedTime(recorded, unrecorded), std::logic_error);
    CHECK_THROWS_AS(onHost::getElapsedTime(unrecorded, unrecorded), std::logic_error);
}

TEMPLATE_LIST_TEST_CASE("queue-created timing-enabled events measure queued work", "", TestApis)
{
    auto [device, executor] = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto start = queue.makeEvent(TestEventPolicy{});
    auto end = queue.makeEvent();
    onHost::concepts::FrameSpec auto const frameSpec = onHost::getFrameSpec(device, executor, Vec{1u});

    static_assert(ALPAKA_TYPEOF(start.getTiming()){} == timing::enabled);
    static_assert(ALPAKA_TYPEOF(end.getTiming()){} == timing::enabled);
    CHECK(start.getPolicyList().hasPolicy(TestEventPolicy{}));

    queue.enqueue(start);
    queue.enqueue(frameSpec, KernelBundle{EmptyTimingKernel{}});
    queue.enqueue(end);

    auto const elapsed = onHost::getElapsedTime(start, end);
    CHECK(elapsed >= std::chrono::duration<double>::zero());
}

TEMPLATE_LIST_TEST_CASE("timing-enabled events include host tasks", "", TestApis)
{
    auto [device, executor] = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto start = device.makeEvent(timing::enabled);
    auto end = device.makeEvent(timing::enabled);
    onHost::concepts::FrameSpec auto const frameSpec = onHost::getFrameSpec(device, executor, Vec{1u});
    constexpr auto hostTaskDuration = std::chrono::seconds{1};

    queue.enqueue(start);
    queue.enqueue(frameSpec, KernelBundle{EmptyTimingKernel{}});
    queue.enqueueHostFn([hostTaskDuration]() { std::this_thread::sleep_for(hostTaskDuration); });
    queue.enqueue(frameSpec, KernelBundle{EmptyTimingKernel{}});
    queue.enqueue(end);

    auto const elapsed = onHost::getElapsedTime(start, end);
    CHECK(elapsed > hostTaskDuration);
}

TEMPLATE_LIST_TEST_CASE("basic queue wait for event", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());

    onHost::Queue queue0 = device.makeQueue();
    onHost::Queue queue1 = device.makeQueue();
    onHost::Event ev = device.makeEvent();
    CHECK(ev.isComplete() == true);
    queue0.enqueue(ev);
    onHost::wait(ev);
    CHECK(ev.isComplete() == true);
    queue0.enqueueHostFn([]() { std::this_thread::sleep_for(std::chrono::milliseconds(100u)); });
    queue0.enqueue(ev);
    queue1.waitFor(ev);
    onHost::wait(queue1);
    CHECK(ev.isComplete() == true);
}

TEMPLATE_LIST_TEST_CASE("wait for event enqueued concurrently on a queue with a long-running host task", "", TestApis)
{
    auto testSingleQueueKind = [&](auto queueKind)
    {
        DYNAMIC_SECTION("Ran with the following queueKind: " << alpaka::onHost::getName(queueKind))
        {
            onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());

            onHost::Queue producerQueue = device.makeQueue(queueKind);
            onHost::Queue consumerQueue = device.makeQueue(queueKind);
            onHost::Event event = device.makeEvent();
            std::atomic_bool hostTaskDoneSignal = false;
            std::promise<void> hostTaskStartSignalPromise;
            auto hostTaskStartSignalFuture = hostTaskStartSignalPromise.get_future();
            std::jthread hostTaskEnqueueThread(
                [&]
                {
                    producerQueue.enqueueHostFn(
                        [&]
                        {
                            // signal to the outside host thread, that the enqueued host function has started
                            hostTaskStartSignalPromise.set_value();
                            // some duration of waiting say 100ms sleep
                            // set some boolean global flag
                            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                            hostTaskDoneSignal.store(true);
                        });
                });
            // make sure that the long-running task was actually started
            hostTaskStartSignalFuture.wait();
            // if the implementation is correct the producerQueue has to block until hostTaskDoneSignal is true
            producerQueue.enqueue(event);
            consumerQueue.waitFor(event);
            onHost::wait(consumerQueue);
            REQUIRE(hostTaskDoneSignal.load());
        }
        return 0;
    };
    onHost::executeForEach(testSingleQueueKind, testQueueKinds);
}

TEMPLATE_LIST_TEST_CASE("wait for event preserves consumer queue order", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    INFO(device.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SKIP(
            "Event tests can not be executed with " << device.getName()
                                                    << " because the device does not support concurrent queues.");
    }
    if(device.getApi() == api::oneApi && device.getDeviceKind() == deviceKind::intelGpu)
    {
        SKIP("Skip test for " << device.getName() << " because the test is typically deadlocking.");
    }

    auto testSingleQueueKind = [&](auto queueKind)
    {
        DYNAMIC_SECTION("Ran with the following queueKind: " << alpaka::onHost::getName(queueKind))
        {
            alpaka::onHost::Queue producerQueue = device.makeQueue(queueKind::nonBlocking);
            alpaka::onHost::Queue consumerQueue = device.makeQueue(queueKind);

            auto produceKernel = TriggerKernel{device};
            onHost::Event event = device.makeEvent();

            std::promise<void> releaseEventPromise;
            auto releaseEventFuture = releaseEventPromise.get_future().share();
            // producer queue must be non-blocking
            produceKernel.submit(producerQueue);

            producerQueue.enqueue(event);

            std::atomic_bool consumerTaskDoneSignal = false;
            std::promise<void> consumerTaskStartSignalPromise;
            auto consumerTaskStartSignalFuture = consumerTaskStartSignalPromise.get_future();
            std::jthread consumerTaskEnqueueThread(
                [&]
                {
                    consumerQueue.enqueueHostFn(
                        [&]
                        {
                            consumerTaskStartSignalPromise.set_value();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                            consumerTaskDoneSignal.store(true);
                        });
                });

            consumerTaskStartSignalFuture.wait();
            std::jthread eventReleaseThread(
                [&]
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10u));
                    produceKernel.trigger();
                });

            // Waiting for the event must not overtake the consumer queue's preceding host task.
            consumerQueue.waitFor(event);
            if constexpr(queueKind.isBlocking())
                REQUIRE(consumerTaskDoneSignal.load());
            else
            {
                std::promise<bool> observerTaskResultPromise;
                auto observerTaskResultFuture = observerTaskResultPromise.get_future();
                consumerQueue.enqueueHostFn([&]
                                            { observerTaskResultPromise.set_value(consumerTaskDoneSignal.load()); });
                REQUIRE(observerTaskResultFuture.get());
            }
        }
        return 0;
    };
    onHost::executeForEach(testSingleQueueKind, testQueueKinds);
}

TEMPLATE_LIST_TEST_CASE("deferred host task preserves queue order", "", TestApis)
{
    auto testSingleQueueKind = [&](auto queueKind)
    {
        DYNAMIC_SECTION("Ran with the following queueKind: " << alpaka::onHost::getName(queueKind))
        {
            onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
            onHost::Queue queue = device.makeQueue(queueKind);

            std::atomic_bool hostTaskDoneSignal = false;
            std::promise<void> hostTaskStartSignalPromise;
            auto hostTaskStartSignalFuture = hostTaskStartSignalPromise.get_future();
            std::jthread hostTaskEnqueueThread(
                [&]
                {
                    queue.enqueueHostFn(
                        [&]
                        {
                            // Ensure that the deferred task is enqueued while this task is still running.
                            hostTaskStartSignalPromise.set_value();
                            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                            hostTaskDoneSignal.store(true);
                        });
                });

            hostTaskStartSignalFuture.wait();
            std::promise<bool> deferredTaskResultPromise;
            auto deferredTaskResultFuture = deferredTaskResultPromise.get_future();
            // A deferred host task may execute after later queue operations, but never before an earlier task.
            // In particular, a blocking queue must finish the running task before submitting this callback.
            queue.enqueueHostFnDeferred([&] { deferredTaskResultPromise.set_value(hostTaskDoneSignal.load()); });

            REQUIRE(deferredTaskResultFuture.get());
        }
        return 0;
    };
    onHost::executeForEach(testSingleQueueKind, testQueueKinds);
}

TEMPLATE_LIST_TEST_CASE("test trigger event", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{cfg};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SUCCEED(
            "Event tests can not be executed with " << deviceSpec.getName()
                                                    << " because the device does not support concurrent queues.");
        return;
    }

    onHost::Queue queue = device.makeQueue();
    auto k = TriggerKernel{device};
    k.submit(queue);
    // we will deadlock here in case the GPU cannot see the state change
    k.trigger();
    k.wait();
    CHECK(k.assumeComplete());
}

TEMPLATE_LIST_TEST_CASE("eventTestShouldBeFalseWhileInQueueAndTrueAfterBeingProcessed", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{cfg};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SUCCEED(
            "Event tests can not be executed with " << deviceSpec.getName()
                                                    << " because the device does not support concurrent queues.");
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    auto k = TriggerKernel{device};

    k.submit(q1);
    REQUIRE(k.assumeNotComplete());

    k.trigger();
    k.wait();
    REQUIRE(k.assumeComplete());
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfNobodyWaitsFor", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{cfg};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SUCCEED(
            "Event tests can not be executed with " << deviceSpec.getName()
                                                    << " because the device does not support concurrent queues.");
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        SUCCEED("Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking.");
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    auto k1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    REQUIRE(k1.assumeComplete());

    k1.submit(q1);
    // wait to detect if the kernel ends before we trigger the end
    std::this_thread::sleep_for(std::chrono::milliseconds(500u));
    // q1 = [k1]
    REQUIRE(k1.assumeNotComplete());

    q1.enqueue(e1);
    // q1 = [k1, e1]
    REQUIRE(k1.assumeNotComplete());
    REQUIRE(!e1.isComplete());

    k2.submit(q1);
    // q1 = [k1, e1, k2]
    REQUIRE(k1.assumeNotComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(k2.assumeNotComplete());

    // re-enqueue should be possible
    q1.enqueue(e1);
    // q1 = [k1, k2, e1]
    REQUIRE(k1.assumeNotComplete());
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(!e1.isComplete());

    k1.trigger();
    // q1 = [k2, e1]
    REQUIRE(k1.assumeComplete());
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(!e1.isComplete());

    k2.trigger();
    // q1 = [e1]
    REQUIRE(k2.assumeComplete());
    onHost::wait(e1);
    REQUIRE(e1.isComplete());
}

TEMPLATE_LIST_TEST_CASE("eventReEnqueueShouldBePossibleIfSomeoneWaitsFor", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{cfg};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SUCCEED(
            "Event tests can not be executed with " << deviceSpec.getName()
                                                    << " because the device does not support concurrent queues.");
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        SUCCEED("Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking.");
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    auto k1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    auto e2 = device.makeEvent();

    k1.submit(q1);
    // q1 = [k1]
    REQUIRE(k1.assumeNotComplete());

    q1.enqueue(e1);
    // q1 = [k1, e1]
    REQUIRE(k1.assumeNotComplete());
    REQUIRE(!e1.isComplete());

    k2.submit(q1);
    // q1 = [k1, e1, k2]
    REQUIRE(k1.assumeNotComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(k2.assumeNotComplete());

    // wait for e1
    q2.waitFor(e1);
    // q2 = [->e1]

    q2.enqueue(e2);
    // q2 = [->e1, e2]
    REQUIRE(!e2.isComplete());

    // re-enqueue should be possible
    q1.enqueue(e1);
    // q1 = [k1, e1, k2, e1_new]
    // q2 = [->e1, e2]
    REQUIRE(k1.assumeNotComplete());
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());

    k1.trigger();
    // q1 = [k2, e1_new]
    // q2 = []
    REQUIRE(k1.assumeComplete());
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(e2.isComplete());

    k2.trigger();
    // q1 = []
    // q2 = []
    REQUIRE(k2.assumeComplete());
    onHost::wait(e1);
    REQUIRE(e1.isComplete());
    onHost::wait(e2);
    REQUIRE(e2.isComplete());
}

TEMPLATE_LIST_TEST_CASE("waitForEventThatAlreadyFinishedShouldBeSkipped", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{cfg};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SUCCEED(
            "Event tests can not be executed with " << deviceSpec.getName()
                                                    << " because the device does not support concurrent queues.");
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        SUCCEED("Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking.");
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    auto k1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();

    // 1. kernel k1 is enqueued into queue q1
    k1.submit(q1);
    // q1 = [k1]

    // 2. event e1 is enqueued into queue q1
    q1.enqueue(e1);
    // q1 = [k1, e1]

    // 3. kernel k2 is enqueued into queue q2
    k2.submit(q2);
    // q2 = [k2]

    // 4. q2 waits for e1
    q2.waitFor(e1);
    // q2 = [k2, ->e1]

    // 5. kernel k1 finishes
    k1.trigger();
    // q1 = [e1]
    // q2 = [k2, ->e1]

    // 6. e1 is finished
    onHost::wait(e1);
    // q1 = []
    // q2 = [k2, ->e1]
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(e1.isComplete());

    // 7. e1 is re-enqueued again but this time into q2
    q2.enqueue(e1);

    // q2 = [k2, ->e1, e1]
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(!e1.isComplete());

    // 8. k2 is triggered
    k2.trigger();
    // q2 = [e1]
    REQUIRE(k2.assumeComplete());

    // 9. e1 had already been signaled, so there should not be waited even though the event is now reused within
    // q2 and its current state is 'unfinished' again. q2 = [e1]

    // Both queues should successfully finish
    onHost::wait(q1);
    onHost::wait(q2);
}

TEMPLATE_LIST_TEST_CASE("evReEnqueueWithSomeoneWaitsForEventInOrderLifetimeRelease", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{cfg};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SUCCEED(
            "Event tests can not be executed with " << deviceSpec.getName()
                                                    << " because the device does not support concurrent queues.");
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        SUCCEED("Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking.");
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    onHost::Queue q3 = device.makeQueue();
    auto k1_0 = TriggerKernel{device};
    auto k1_1 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto k3 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    auto e2 = device.makeEvent();
    auto e3 = device.makeEvent();

    k1_0.submit(q1);
    // q1 = [k1_0]
    q1.enqueue(e1);
    // q1 = [k1_0, e1]
    k2.submit(q2);
    // q2 = [k2]
    REQUIRE(k1_0.assumeNotComplete());

    q2.waitFor(e1);
    // q2 = [k2,->e1]
    q2.enqueue(e2);
    // q2 = [k2,->e1, e2]
    k1_1.submit(q1);
    q1.enqueue(e1);
    // q1 = [k1_0, e1, k1_1, e1_new]
    k3.submit(q3);
    q3.waitFor(e1);
    // q3 = [k3, ->e1]
    q3.enqueue(e3);
    // q3 = [k3, ->e1, e3]

    // q1 = [k1_0,e1,k1_1,e1_new]
    // q2 = [k2,->e1,e2]
    // q3 = [k3,->e1_new,e3]
    REQUIRE(k1_0.assumeNotComplete());
    REQUIRE(k1_1.assumeNotComplete());
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(k3.assumeNotComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());

    k3.trigger();
    // q1 = [k1_0,e1,k1_1,e1_new]
    // q2 = [k2,->e1,e2]
    // q3 = [->e1_new,e3]
    REQUIRE(k1_0.assumeNotComplete());
    REQUIRE(k1_1.assumeNotComplete());
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(k3.assumeComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());

    k2.trigger();
    // q1 = [k1_0,e1,k1_1,e1_new]
    // q2 = [->e1,e2]
    // q3 = [->e1_new,e3]
    REQUIRE(k1_0.assumeNotComplete());
    REQUIRE(k1_1.assumeNotComplete());
    REQUIRE(k2.assumeComplete());
    REQUIRE(k3.assumeComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());

    // After the kernel k1_0 is released e3 is not allowed to be ready because q3 depends on the oldest e1_new
    // state.
    k1_0.trigger();
    // q1 = [k1_1,e1_new]
    // q2 = []
    // q3 = [->e1_new,e3]
    REQUIRE(k1_0.assumeComplete());
    REQUIRE(k1_1.assumeNotComplete());
    REQUIRE(k2.assumeComplete());
    REQUIRE(k3.assumeComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(!e3.isComplete());

    k1_1.trigger();
    // q1 = []
    // q2 = []
    // q3 = []
    REQUIRE(k1_0.assumeComplete());
    REQUIRE(k1_1.assumeComplete());
    REQUIRE(k2.assumeComplete());
    REQUIRE(k3.assumeComplete());
    REQUIRE(e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(e3.isComplete());
}

TEMPLATE_LIST_TEST_CASE("EventOutOfOrderLifetimeRelease", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{cfg};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    bool hasConcurrentKernelQueues = checkIfDeviceCanExecuteEventTests(device);
    if(!hasConcurrentKernelQueues)
    {
        /* We cannot execute the event tests with OneApi on Intel GPU because the emulated kernel trigger via another
         * kernel in a separate queue. The second reason is that kernel, memory operation enqueued in different queues
         * will not run out of order which is assumed for some of the tests.
         */
        SUCCEED(
            "Event tests can not be executed with " << deviceSpec.getName()
                                                    << " because the device does not support concurrent queues.");
        return;
    }
    if(deviceSpec.getApi() == api::oneApi && deviceSpec.getDeviceKind() == deviceKind::intelGpu)
    {
        SUCCEED("Skip test for " << deviceSpec.getName() << " because the test is typically deadlocking.");
        return;
    }

    onHost::Queue q1 = device.makeQueue();
    onHost::Queue q2 = device.makeQueue();
    onHost::Queue q3 = device.makeQueue();
    auto k1_0 = TriggerKernel{device};
    auto k2 = TriggerKernel{device};
    auto e1 = device.makeEvent();
    auto e2 = device.makeEvent();
    auto e3 = device.makeEvent();

    k1_0.submit(q1);
    // q1 = [k1_0]
    q1.enqueue(e1);
    // q1 = [k1_0, e1]
    k2.submit(q2);
    // q2 = [k2]
    REQUIRE(k1_0.assumeNotComplete());

    q3.waitFor(e1);
    // q3 = [->e1]
    q3.enqueue(e3);
    // q3 = [->e1, e3]

    q2.enqueue(e1);
    // q2 = [k2, e1_new]

    q2.enqueue(e2);
    // q1 = [k1_0,e1]
    // q2 = [k2,e1_new,e2]
    // q3 = [->e1,e3]

    REQUIRE(k1_0.assumeNotComplete());
    REQUIRE(k2.assumeNotComplete());
    REQUIRE(!e1.isComplete());
    REQUIRE(!e2.isComplete());
    REQUIRE(!e3.isComplete());


    // We release first the kernel which is blocking the most recent enqueue of event e1.
    // q3 is not allowed to be freed because this queue depends on the oldest enqueue of e1.
    k2.trigger();
    // q1 = [k1_0,e1]
    // q2 = []
    // q3 = [->e1,e3]

    REQUIRE(k1_0.assumeNotComplete());
    REQUIRE(k2.assumeComplete());
    REQUIRE(e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(!e3.isComplete());

    k1_0.trigger();
    // q1 = []
    // q2 = []
    // q3 = []

    REQUIRE(k1_0.assumeComplete());
    REQUIRE(k2.assumeComplete());
    REQUIRE(e1.isComplete());
    REQUIRE(e2.isComplete());
    REQUIRE(e3.isComplete());

    onHost::wait(q1);
    onHost::wait(q2);
    onHost::wait(q3);
}
