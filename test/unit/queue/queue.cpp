/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Test-only tag verifying that applications can register additional queue policies. */
struct TestQueuePolicy
{
};

template<>
struct alpaka::trait::IsQueuePolicy<TestQueuePolicy> : std::true_type
{
};

static_assert(alpaka::concepts::QueuePolicy<TestQueuePolicy>);

constexpr auto defaultQueuePolicies = onHost::QueuePolicyList{};
static_assert(alpaka::concepts::QueuePolicyList<decltype(defaultQueuePolicies)>);
static_assert(defaultQueuePolicies.getQueueKind() == queueKind::nonBlocking);
static_assert(defaultQueuePolicies.getTiming() == timing::disabled);

constexpr auto testQueuePolicies = onHost::QueuePolicyList{queueKind::blocking, timing::enabled, TestQueuePolicy{}};
static_assert(alpaka::concepts::QueuePolicyList<decltype(testQueuePolicies)>);
static_assert(!alpaka::concepts::QueuePolicyList<TestQueuePolicy>);
static_assert(testQueuePolicies.getQueueKind() == queueKind::blocking);
static_assert(testQueuePolicies.getTiming() == timing::enabled);
static_assert(testQueuePolicies.hasPolicy(TestQueuePolicy{}));
static_assert(!testQueuePolicies.hasPolicy(queueKind::nonBlocking));

struct NoArgumentsKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const&) const
    {
    }
};

/** test that we can execute a kernel which has no arguments
 *
 * All arguments could be stored e.g. as members.
 */
TEMPLATE_LIST_TEST_CASE("kernel no arguments", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    INFO("mem=" << device.getDeviceProperties().globalMemCapacityBytes);
    INFO("free mem=" << device.getFreeGlobalMemBytes());

    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    queue.enqueue(onHost::FrameSpec{1, 1, exec}, KernelBundle{NoArgumentsKernel{}});

    // to avoid to trigger Catch2 NoAssert check
    STATIC_REQUIRE(true);
}

struct IotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        // check that frame extent keeps compile time const-ness
        static_assert(alpaka::concepts::CVector<ALPAKA_TYPEOF(acc[layer::thread].count())>);
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            // Debug-only trace removed in favor of Catch2 diagnostics.
            out[i.x()] = i.x();
        }
    }
};

TEMPLATE_LIST_TEST_CASE("iota", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{12u};
    auto dBuff = onHost::alloc<uint32_t>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    queue.enqueue(onHost::FrameSpec{extent / frameSize, frameSize, exec}, KernelBundle{IotaKernel{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx.x() == hBuff[idx]); });
}

struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = i;
        }
    }
};

TEMPLATE_LIST_TEST_CASE("iota2D", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr Vec extent = Vec{8u, 16u};
    auto dBuff = onHost::alloc<Vec<uint32_t, 2u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{2u, 4u};
    queue.enqueue(onHost::FrameSpec{extent / frameSize, frameSize, exec}, KernelBundle{IotaKernelND{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}

TEMPLATE_LIST_TEST_CASE("iota3D", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr Vec extent = Vec{4u, 8u, 16u};
    auto dBuff = onHost::alloc<Vec<uint32_t, 3u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{2u, 4u, 8u};
    queue.enqueue(onHost::FrameSpec{extent / frameSize, frameSize, exec}, KernelBundle{IotaKernelND{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}

TEMPLATE_LIST_TEST_CASE("iota4D", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr Vec extent = Vec{4u, 8u, 16, 32};
    auto dBuff = onHost::alloc<Vec<uint32_t, 4u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{2u, 4u, 8u, 4u};
    queue.enqueue(onHost::FrameSpec{extent / frameSize, frameSize, exec}, KernelBundle{IotaKernelND{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}

/** Test index container dimension slicing
 *
 * @todo This is a very bad example for slicing. Find a better useful example.
 */
template<typename T_Selection>
struct IotaKernelNDSelection
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, alpaka::concepts::Vector auto chunkSize) const
    {
        alpaka::concepts::Vector auto outExtents = out.getExtents();
        // select z dimension only and iterate over y,x in the inner loops
        for(auto dataOffset :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{outExtents})[CVec<uint32_t, 0u>{}])
        {
            // dataOffset is still 3-dimensional but contains 0 in y,x
            static_assert(alpaka::concepts::Dim<ALPAKA_TYPEOF(dataOffset), 3>);
            // All threads in a block now iterates over the dimensions sliced out in the for loop above.
            for(auto fullDataIdx :
                onAcc::makeIdxMap(acc, onAcc::worker::allThreads, IdxRange{dataOffset, outExtents})[T_Selection{}])
            {
                for(auto elemIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                    // only elemIdx 1 is applying the change
                    if(linearize(chunkSize, elemIdx) == 1u)
                    {
                        // use atomics to detect data races where more than one thread is updating the result
                        onAcc::atomicAdd(acc, &(out[fullDataIdx][0]), fullDataIdx[0]);
                        onAcc::atomicAdd(acc, &(out[fullDataIdx][1]), fullDataIdx[1]);
                        onAcc::atomicAdd(acc, &(out[fullDataIdx][2]), fullDataIdx[2]);
                    }
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("iota3D 2D iterate", "", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr Vec numBlocks = Vec{4u, 8u, 16u};
    auto numBlocksReduced = numBlocks;

    // Within the kernel we will iterate over y, x dimension
    [[maybe_unused]] auto selection = CVec<uint32_t, 2u, 1u>{};
    // start a 3D extent but with size 1 in y, x dimension
    numBlocksReduced.ref(selection) = 1u;

    INFO("numBlocksReduced=" << numBlocksReduced);
    auto dBuff = onHost::alloc<Vec<uint32_t, 3u>>(device, numBlocks);

    auto hBuff = onHost::allocHostLike(dBuff);
    memset(queue, dBuff, 0u);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{1u, 1u, 2u};

    /* Use a frame specification where we have frames in z only.
     * The kernel is later responsible that each data point of the output is filled with data.
     */
    auto frameSpec = onHost::FrameSpec{numBlocksReduced, frameSize, exec};

    queue.enqueue(frameSpec, KernelBundle{IotaKernelNDSelection<ALPAKA_TYPEOF(selection)>{}, dBuff, frameSize});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(numBlocks, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}

/** ViewWrapper for validating the memcpy and memset interfaces.
 *
 * The wrapper is disabling the copy and move operations to within alpaka.
 */
template<alpaka::concepts::IMdSpan T_Span>
struct TestMdSpan
{
    using value_type = typename T_Span::value_type;

    constexpr TestMdSpan(T_Span const& span) : m_span(span)
    {
    }

    constexpr TestMdSpan() = delete;
    constexpr TestMdSpan(TestMdSpan const&) = delete;
    constexpr TestMdSpan& operator=(TestMdSpan const&) = delete;
    constexpr TestMdSpan(TestMdSpan&&) = delete;
    constexpr TestMdSpan& operator=(TestMdSpan&&) = delete;

    constexpr auto* data() const
    {
        return m_span.data();
    }

    constexpr auto* data()
    {
        return m_span.data();
    }

    constexpr auto getExtents() const
    {
        return m_span.getExtents();
    }

    constexpr auto getPitches() const
    {
        return m_span.getPitches();
    }

    constexpr auto getApi() const
    {
        return alpaka::getApi(m_span);
    }

private:
    T_Span m_span;
};

/** Test that memcpy and memset can be called with non copy-able and move-able data as lvalue and rvalue. */
TEMPLATE_LIST_TEST_CASE("memcpy", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    onHost::Queue queue = device.makeQueue();
    constexpr Vec problemSize = Vec{16u};

    auto dBuff = onHost::alloc<size_t>(device, problemSize);
    auto hBuff = onHost::allocHostLike(dBuff);

    // test rvalue
    onHost::memcpy(queue, TestMdSpan{dBuff.getView()}, TestMdSpan{hBuff.getView()});

    // check fill
    onHost::fill(queue, TestMdSpan{dBuff.getView()}, size_t{42u});
    onHost::memcpy(queue, TestMdSpan{hBuff.getView()}, TestMdSpan{dBuff.getView()});
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{42}); });

    // check memset
    onHost::memset(queue, TestMdSpan{dBuff.getView()}, 0u);
    onHost::memcpy(queue, TestMdSpan{hBuff.getView()}, TestMdSpan{dBuff.getView()});
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{0}); });

    size_t refence = 0u;
    for(auto& v : hBuff)
        v = refence++;

    onHost::memcpy(queue, TestMdSpan{dBuff.getView()}, TestMdSpan{hBuff.getView()});
    onHost::wait(queue);

    // overwrite host values to be sure that memcpy later on updates the host values
    for(auto& v : hBuff)
        v = 42;

    // test lvalue
    auto dlvalue = TestMdSpan{dBuff.getView()};
    auto hlvalue = TestMdSpan{hBuff.getView()};
    onHost::memcpy(queue, hlvalue, dlvalue);
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == linearize(problemSize, idx)); });

    // check fill
    onHost::fill(queue, dlvalue, size_t{42u});
    onHost::memcpy(queue, hlvalue, dlvalue);
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{42}); });

    // check memset
    onHost::memset(queue, dlvalue, 0u);
    onHost::memcpy(queue, hlvalue, dlvalue);
    onHost::wait(queue);

    // validate without using the forward iterator
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{0}); });
}

TEMPLATE_LIST_TEST_CASE("host task callback", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    onHost::Queue queue = device.makeQueue();

    std::promise<bool> promise;

    queue.enqueueHostFn([&]() { promise.set_value(true); });

    INFO("Host callback enqueued");
    CHECK(promise.get_future().get());
}

TEMPLATE_LIST_TEST_CASE("host task", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    onHost::Queue queue = device.makeQueue();

    bool flag = false;
    auto task = [&] { flag = true; };
    queue.enqueueHostFn(task); // lvalue
    onHost::wait(queue);
    CHECK(flag == true);

    flag = false;
    queue.enqueueHostFn(std::move(task)); // xvalue
    onHost::wait(queue);
    CHECK(flag == true);

    flag = false;
    queue.enqueueHostFn([&] { flag = true; }); // prvalue
    onHost::wait(queue);
    CHECK(flag == true);
}

TEMPLATE_LIST_TEST_CASE("queue wait should work", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    onHost::Queue queue = device.makeQueue();

    std::atomic<bool> callbackFinished{false};
    queue.enqueueHostFn(
        [&callbackFinished]() noexcept
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            callbackFinished = true;
        });

    onHost::wait(queue);
    CHECK(callbackFinished.load() == true);
}

TEMPLATE_LIST_TEST_CASE("task is destroyed after execution", "", TestApis)
{
    onHost::Device device = test::getDeviceOrSkipTest(TestType::makeDict());
    onHost::Queue queue = device.makeQueue();

    struct Task
    {
        std::atomic<bool>* destroyed;

        explicit Task(std::atomic<bool>& a) : destroyed(&a)
        {
        }

        Task(Task const&) = default;

        ~Task()
        {
            *destroyed = true;
        }

        void operator()() const
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1u));
        }
    };

    std::atomic<bool> destroyed{false};
    queue.enqueueHostFn(Task{destroyed});

    onHost::wait(queue);
    CHECK(destroyed == true);
}
