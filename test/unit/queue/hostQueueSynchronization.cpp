/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <thread>

using namespace alpaka;

namespace
{
    // A timeout here means the operation is correctly blocked by our gate.
    // CTest's separate timeout detects failure to finish after releasing it.
    // The started promises precede API entry; this delay only encourages the interleaving.
    constexpr auto blockingInterval = std::chrono::milliseconds{100};
    // waitStarted is set before waitFor registers the dependency; poll briefly until that registration is visible.
    constexpr auto waitRegistrationTimeout = std::chrono::milliseconds{1000};
} // namespace

TEST_CASE("host blocking event wait allows concurrent re-recording", "[queue]")
{
    // Producer: gate -> E1. Consumer: wait(E1) -> E2, using the same event object.
    // Re-recording must leave the existing wait dependent on E1, not E2.
    auto device = onHost::makeHostDevice();
    auto producer = device.makeQueue(queueKind::nonBlocking);
    auto consumer = device.makeQueue(queueKind::blocking);
    auto event = device.makeEvent();

    std::promise<void> releaseProducer;
    auto producerGate = releaseProducer.get_future().share();
    producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    producer.enqueue(event);

    std::promise<void> waitStarted;
    auto waiter = std::async(
        std::launch::async,
        [&]
        {
            waitStarted.set_value();
            consumer.waitFor(event);
        });
    waitStarted.get_future().wait();
    // waitStarted only precedes API entry. Wait until waitFor published its pending dependency.
    auto const waitRegistrationDeadline = std::chrono::steady_clock::now() + waitRegistrationTimeout;
    // wait until the consumer wait is registered
    while(consumer.isEmpty() && std::chrono::steady_clock::now() < waitRegistrationDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});

    std::promise<void> recordFinished;
    auto recordFinishedFuture = recordFinished.get_future();

    std::promise<void> recordStarted;
    auto recorder = std::async(
        std::launch::async,
        [&]
        {
            recordStarted.set_value();
            consumer.enqueue(event);
            recordFinished.set_value();
        });
    recordStarted.get_future().wait();
    // Ensure that the event is re-enqueued before the producer is unlocked.
    auto const recordStatus = recordFinishedFuture.wait_for(blockingInterval);
    CHECK(recordStatus == std::future_status::ready);
    auto const waitStatus = waiter.wait_for(blockingInterval);
    CHECK(waitStatus == std::future_status::timeout);

    // event is blocked until here because producer host function is waiting on producerGate/releaseProducer
    releaseProducer.set_value();
    // Releasing the gate must allow E1 -> consumer wait -> E2 to finish.
    // Regression: recorder holds the event lock while waiting for the consumer
    // lock; the consumer holds its lock while E1 needs the event lock to finish.
    waiter.get();
    recorder.get();
    onHost::wait(event);
    CHECK(event.isComplete());
}

TEST_CASE("host blocking event wait keeps the queue nonempty", "[queue]")
{
    // Producer: gate -> event. Consumer: wait(event), with no other work.
    // An unresolved dependency still counts as unfinished queue work. If waitFor
    // bypasses the busy flag, isEmpty() and wait(queue) incorrectly report completion.
    auto device = onHost::makeHostDevice();
    auto producer = device.makeQueue(queueKind::nonBlocking);
    auto consumer = device.makeQueue(queueKind::blocking);
    auto event = device.makeEvent();

    std::promise<void> releaseProducer;
    auto producerGate = releaseProducer.get_future().share();
    producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    producer.enqueue(event);

    std::promise<void> waitStarted;
    auto eventWaiter = std::async(
        std::launch::async,
        [&]
        {
            waitStarted.set_value();
            consumer.waitFor(event);
        });
    waitStarted.get_future().wait();
    // waitStarted only precedes API entry. Wait until waitFor published its pending dependency.
    auto const waitRegistrationDeadline = std::chrono::steady_clock::now() + waitRegistrationTimeout;
    while(consumer.isEmpty() && std::chrono::steady_clock::now() < waitRegistrationDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    auto const eventWaitStatus = eventWaiter.wait_for(blockingInterval);
    CHECK(eventWaitStatus == std::future_status::timeout);
    bool const emptyWhileWaiting = consumer.isEmpty();
    CHECK_FALSE(emptyWhileWaiting);

    // Queue synchronization must stay blocked for the same dependency.
    auto queueWaiter = std::async(std::launch::async, [&] { onHost::wait(consumer); });
    auto const queueWaitStatus = queueWaiter.wait_for(blockingInterval);
    releaseProducer.set_value();
    eventWaiter.get();
    queueWaiter.get();

    onHost::wait(event);
    CHECK(queueWaitStatus == std::future_status::timeout);
    CHECK(event.isComplete());
}
