/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Test-only tag verifying that applications can register additional memory allocation policies. */
struct TestMemoryPolicy
{
};

template<>
struct alpaka::trait::IsMemoryPolicy<TestMemoryPolicy> : std::true_type
{
};

static_assert(alpaka::concepts::MemoryPolicy<TestMemoryPolicy>);

constexpr auto defaultMemoryPolicies = onHost::MemoryPolicyList<>{};
static_assert(alpaka::concepts::MemoryPolicyList<decltype(defaultMemoryPolicies)>);
static_assert(defaultMemoryPolicies.getMemoryProperty() == memoryProperty::defaultProperty);

constexpr auto testMemoryPolicies = onHost::MemoryPolicyList{memoryProperty::bestBandwidth, TestMemoryPolicy{}};
static_assert(alpaka::concepts::MemoryPolicyList<decltype(testMemoryPolicies)>);
static_assert(!alpaka::concepts::MemoryPolicyList<TestMemoryPolicy>);
static_assert(testMemoryPolicies.getMemoryProperty() == memoryProperty::bestBandwidth);
static_assert(testMemoryPolicies.hasPolicy(TestMemoryPolicy{}));
static_assert(!testMemoryPolicies.hasPolicy(memoryProperty::bestLatency));

/** Test-only device which declares no memory property support at all. */
struct TestDeviceWithoutMemoryProperties
{
};

/** Test-only device which implements memoryProperty::Locality only. */
struct TestDeviceWithLocality
{
};

template<>
struct alpaka::onHost::trait::IsMemoryPropertySupportedBy::Op<memoryProperty::Locality, TestDeviceWithLocality>
    : std::true_type
{
};

// A placement preference must be opt-in per device, memoryProperty::Default is always available.
static_assert(
    !onHost::isMemoryPropertySupportedBy_v<memoryProperty::BestBandwidth, TestDeviceWithoutMemoryProperties>);
static_assert(!onHost::isMemoryPropertySupportedBy_v<memoryProperty::BestLatency, TestDeviceWithoutMemoryProperties>);
static_assert(!onHost::isMemoryPropertySupportedBy_v<memoryProperty::Locality, TestDeviceWithoutMemoryProperties>);
static_assert(onHost::isMemoryPropertySupportedBy_v<memoryProperty::Default, TestDeviceWithoutMemoryProperties>);
static_assert(
    std::tuple_size_v<ALPAKA_TYPEOF(onHost::supportedMemoryProperties(TestDeviceWithoutMemoryProperties{}))> == 1u);

static_assert(onHost::isMemoryPropertySupportedBy_v<memoryProperty::Locality, TestDeviceWithLocality>);
static_assert(!onHost::isMemoryPropertySupportedBy_v<memoryProperty::BestBandwidth, TestDeviceWithLocality>);

// hasProperty() scans left to right and returns the first property the device implements, properties belonging to
// another kind of device are skipped.
static_assert(
    onHost::hasProperty(
        TestDeviceWithLocality{},
        memoryProperty::bestBandwidth,
        memoryProperty::locality,
        memoryProperty::defaultProperty)
    == memoryProperty::locality);

// The leftmost supported property wins, even when a later one is supported too.
static_assert(
    onHost::hasProperty(TestDeviceWithLocality{}, memoryProperty::locality, memoryProperty::defaultProperty)
    == memoryProperty::locality);

// Nothing in the list is supported, the universally available default is returned.
static_assert(
    onHost::hasProperty(
        TestDeviceWithoutMemoryProperties{},
        memoryProperty::bestBandwidth,
        memoryProperty::locality,
        memoryProperty::defaultProperty)
    == memoryProperty::defaultProperty);

// The fallback is never implicit: a list which the device cannot satisfy is a compile time error, so
//   onHost::hasProperty(TestDeviceWithoutMemoryProperties{}, memoryProperty::bestBandwidth)
// does not compile. Accepting the backend default has to be written down as the trailing entry below.
static_assert(
    onHost::hasProperty(TestDeviceWithoutMemoryProperties{}, memoryProperty::defaultProperty)
    == memoryProperty::defaultProperty);

// hostHasProperty() is usable in a constant expression, which proves it constructs no device: onHost::Device has
// no constexpr destructor, so an expression involving one could never appear here.
static_assert(onHost::hostHasProperty(memoryProperty::defaultProperty) == memoryProperty::defaultProperty);

// memoryProperty::defaultProperty is implemented by every device, so the one element list is accepted.
static_assert(
    onHost::hostHasProperty(memoryProperty::bestBandwidth, memoryProperty::defaultProperty)
    == onHost::hostHasProperty(memoryProperty::bestBandwidth, memoryProperty::defaultProperty));

struct IotaValidate
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IMdSpan<int> auto success, concepts::IMdSpan auto in)
        const
    {
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(in.getExtents())))
        {
            /* Each correct result increases the result by one, this avoids false positives if the kernel is not
             * executed.
             */
            if(in[i] == i)
                onAcc::atomicAdd(acc, &success[0], 1);
        }
    }
};

void validateAccess(auto device, alpaka::concepts::Executor auto exec, concepts::IMdSpan auto deviceAccessibleData)
{
    auto deviceStatus = onHost::alloc<int>(device, 1);
    auto hostStatus = onHost::allocHostLike(deviceStatus);
    auto deviceQueue = device.makeQueue();
    REQUIRE(onHost::isDataAccessible(deviceQueue, deviceAccessibleData) == true);
    onHost::fill(deviceQueue, deviceStatus, 0);
    deviceQueue.enqueue(
        getFrameSpec(deviceQueue.getDevice(), exec, deviceAccessibleData.getExtents()),
        KernelBundle{IotaValidate{}, deviceStatus, deviceAccessibleData});
    onHost::memcpy(deviceQueue, hostStatus, deviceStatus);
    onHost::wait(deviceQueue);
    // if the number of the result not matches the extent, a few results are wrong
    REQUIRE(hostStatus[0] == deviceAccessibleData.getExtents().x());
}

void allocDeferredImplicitWait(auto device, alpaka::concepts::Executor auto exec)
{
    onHost::Queue queue0 = device.makeQueue();

    auto hostDevice = onHost::makeHostDevice();

    int dataSize = 42;

    auto hostBuffer = onHost::allocHost<int>(dataSize);
    auto hostBufferMapped = onHost::allocMapped<int>(device, dataSize);
    auto deviceView = onHost::alloc<int>(device, dataSize);
    auto unifiedView = onHost::allocUnified<int>(device, dataSize);

    REQUIRE(onHost::isDataAccessible(hostDevice, hostBuffer) == true);
    REQUIRE(onHost::isDataAccessible(hostDevice, unifiedView) == true);
    REQUIRE(onHost::isDataAccessible(hostDevice, hostBufferMapped) == true);
    for(int i = 0; i < hostBuffer.getExtents().x(); ++i)
    {
        hostBuffer[i] = i;
        // unified memory must be accessible on the host
        unifiedView[i] = i;
        // is located on the host, so it must be accessible
        hostBufferMapped[i] = i;
    }

    auto deviceQueue = device.makeQueue();
    // check that we can copy from unified memory to device memory
    onHost::memcpy(deviceQueue, deviceView, unifiedView);
    onHost::wait(deviceQueue);

    if(getDeviceKind(device) == deviceKind::cpu || getDeviceKind(device) == deviceKind::numaCpu)
    {
        REQUIRE(onHost::isDataAccessible(device, hostBuffer) == true);
        validateAccess(device, exec, hostBuffer);
    }
    else
        REQUIRE(onHost::isDataAccessible(device, hostBuffer) == false);

    // mapped memory is defined to be accessible on the device
    REQUIRE(onHost::isDataAccessible(device, hostBufferMapped) == true);
    validateAccess(device, exec, hostBufferMapped);

    REQUIRE(onHost::isDataAccessible(device, unifiedView) == true);
    validateAccess(device, exec, unifiedView);

    REQUIRE(onHost::isDataAccessible(device, deviceView) == true);
    validateAccess(device, exec, deviceView);

    REQUIRE(onHost::isDataAccessible(hostDevice, unifiedView) == true);
    validateAccess(hostDevice, exec::cpuSerial, unifiedView);

    // is located on the host, so it must be accessible
    REQUIRE(onHost::isDataAccessible(device, hostBufferMapped) == true);
    validateAccess(hostDevice, exec::cpuSerial, hostBufferMapped);
}

TEMPLATE_LIST_TEST_CASE("alloc", "", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    allocDeferredImplicitWait(device, exec);
}

using TestDeviceSpecs = std::decay_t<decltype(onHost::getDeviceSpecsFor(onHost::enabledApis))>;

TEMPLATE_LIST_TEST_CASE("alloc zero bytes", "", TestDeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    auto hostDevice = onHost::makeHostDevice();

    // test to allocate zero byte memory to validate of the allocation and free works as expected
    int dataSize = 0;

    [[maybe_unused]] auto hostBuffer = onHost::allocHost<int>(dataSize);
    CHECK(hostBuffer.getExtents() == alpaka::Vec{dataSize});
    [[maybe_unused]] auto hostBufferAsync = onHost::allocDeferred<int>(onHost::makeHostDevice().makeQueue(), dataSize);
    CHECK(hostBufferAsync.getExtents() == alpaka::Vec{dataSize});
    [[maybe_unused]] auto hostBufferMapped = onHost::allocMapped<int>(device, dataSize);
    CHECK(hostBufferMapped.getExtents() == alpaka::Vec{dataSize});
    [[maybe_unused]] auto deviceView = onHost::alloc<int>(device, dataSize);
    CHECK(deviceView.getExtents() == alpaka::Vec{dataSize});
    [[maybe_unused]] auto deviceViewAsync = onHost::allocDeferred<int>(device.makeQueue(), dataSize);
    CHECK(deviceViewAsync.getExtents() == alpaka::Vec{dataSize});
    [[maybe_unused]] auto unifiedView = onHost::allocUnified<int>(device, dataSize);
    CHECK(unifiedView.getExtents() == alpaka::Vec{dataSize});
}

TEMPLATE_LIST_TEST_CASE("alloc with memory property", "", TestDeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    int dataSize = 42;

    // memoryProperty::defaultProperty is supported by every device, therefore it never needs a guard.
    static_assert(onHost::isMemoryPropertySupportedBy(memoryProperty::defaultProperty, device));
    // A device always reports at least memoryProperty::defaultProperty as supported.
    static_assert(std::tuple_size_v<ALPAKA_TYPEOF(onHost::supportedMemoryProperties(device))> >= 1u);

    auto queue = device.makeQueue();
    // A queue resolves to the memory properties of its device.
    static_assert(
        onHost::hasProperty(queue, memoryProperty::bestBandwidth, memoryProperty::defaultProperty)
        == onHost::hasProperty(device, memoryProperty::bestBandwidth, memoryProperty::defaultProperty));

    // allocHost() takes no device argument, it always allocates on the host CPU device. hostHasProperty() queries
    // that device without constructing one, which properties it implements depends on the hwloc build option.
    [[maybe_unused]] auto hostBuffer = onHost::allocHost<int>(
        dataSize,
        onHost::hostHasProperty(memoryProperty::bestBandwidth, memoryProperty::defaultProperty));
    CHECK(hostBuffer.getExtents() == alpaka::Vec{dataSize});

    // It agrees with hasProperty() on an explicitly constructed host device.
    auto hostDevice = onHost::makeHostDevice();
    static_assert(
        onHost::hostHasProperty(memoryProperty::bestBandwidth, memoryProperty::defaultProperty)
        == onHost::hasProperty(hostDevice, memoryProperty::bestBandwidth, memoryProperty::defaultProperty));
    static_assert(
        onHost::hostHasProperty(memoryProperty::locality, memoryProperty::defaultProperty)
        == onHost::hasProperty(hostDevice, memoryProperty::locality, memoryProperty::defaultProperty));

    [[maybe_unused]] auto hostLikeBuffer = onHost::allocHostLike(
        hostBuffer,
        onHost::hostHasProperty(memoryProperty::locality, memoryProperty::defaultProperty));
    CHECK(hostLikeBuffer.getExtents() == hostBuffer.getExtents());

    // Device allocations use hasProperty() so that the same source compiles for every backend, a property which the
    // device does not implement is rejected at compile time when it is passed directly.
    [[maybe_unused]] auto deviceView = onHost::alloc<int>(
        device,
        dataSize,
        onHost::hasProperty(device, memoryProperty::bestLatency, memoryProperty::defaultProperty));
    CHECK(deviceView.getExtents() == alpaka::Vec{dataSize});

    [[maybe_unused]] auto mappedView = onHost::allocMapped<int>(
        device,
        dataSize,
        onHost::hasProperty(device, memoryProperty::locality, memoryProperty::defaultProperty));
    CHECK(mappedView.getExtents() == alpaka::Vec{dataSize});

    [[maybe_unused]] auto unifiedView = onHost::allocUnified<int>(device, dataSize, memoryProperty::defaultProperty);
    CHECK(unifiedView.getExtents() == alpaka::Vec{dataSize});

    [[maybe_unused]] auto likeView = onHost::allocLike(
        device,
        deviceView,
        onHost::hasProperty(device, memoryProperty::bestBandwidth, memoryProperty::defaultProperty));
    CHECK(likeView.getExtents() == deviceView.getExtents());

    // An explicit onHost::MemoryPolicyList.
    [[maybe_unused]] auto explicitPolicyView = onHost::alloc<int>(
        device,
        dataSize,
        onHost::MemoryPolicyList{
            onHost::hasProperty(device, memoryProperty::bestBandwidth, memoryProperty::defaultProperty)});
    CHECK(explicitPolicyView.getExtents() == alpaka::Vec{dataSize});

    [[maybe_unused]] auto deferredView = onHost::allocDeferred<int>(
        queue,
        dataSize,
        onHost::hasProperty(queue, memoryProperty::bestBandwidth, memoryProperty::defaultProperty));
    CHECK(deferredView.getExtents() == alpaka::Vec{dataSize});

    [[maybe_unused]] auto likeDeferredView = onHost::allocLikeDeferred(
        queue,
        deviceView,
        onHost::hasProperty(queue, memoryProperty::locality, memoryProperty::defaultProperty));
    CHECK(likeDeferredView.getExtents() == deviceView.getExtents());
}

/** Evaluates on the host side that all rows start with an address which is a multiple of the alignment of the MdSpan
 *
 * @attention We evaluate device side pointer on the host side, this is ok because we never dereference the pointer and
 * relay on pointer addressing only. If we would have at some point MdSPans where the operator[] is only accessible on
 * the device we need to rewrite this test and perform the evaluations on the compute device.
 *
 * @param data multi-dimensional data which is checked
 */
void validateAlignment(alpaka::concepts::IMdSpan auto data)
{
    using DataType = alpaka::GetValueType_t<ALPAKA_TYPEOF(data)>;
    constexpr uint32_t alignment = alpaka::getAlignment(data).template get<DataType>();
    alpaka::concepts::Vector auto extents = alpaka::onHost::getExtents(data);
    // set the number of columns to 1 to evaluate only the rows
    extents.back() = 1;

    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            auto* rowPtr = &data[idx];
            CHECK((reinterpret_cast<uint64_t>(rowPtr) % alignment) == 0);
        });
}

template<typename T_DataType>
void prepareAlignmentValidation(auto& device, alpaka::concepts::Vector auto extents)
{
    auto hostBuffer = onHost::allocHost<T_DataType>(extents);
    validateAlignment(hostBuffer);
    auto hostBufferAsync = onHost::allocDeferred<T_DataType>(onHost::makeHostDevice().makeQueue(), extents);
    validateAlignment(hostBufferAsync);
    auto hostBufferMapped = onHost::allocMapped<T_DataType>(device, extents);
    validateAlignment(hostBufferMapped);
    auto deviceView = onHost::alloc<T_DataType>(device, extents);
    validateAlignment(deviceView);
    auto deviceViewAsync = onHost::allocDeferred<T_DataType>(device.makeQueue(), extents);
    validateAlignment(deviceViewAsync);
    auto unifiedView = onHost::allocUnified<T_DataType>(device, extents);
    validateAlignment(unifiedView);
}

TEMPLATE_LIST_TEST_CASE("alloc alignment", "", TestDeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    using DataType = int;

    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareAlignmentValidation<DataType>(device, extents), ...); }, extentMdList);
}

template<typename T_DataType>
void volatileBuffers(auto& device, alpaka::concepts::Vector auto extents)
{
    // just test if they can be allocated and destructed again
    auto hostBuffer = onHost::allocHost<T_DataType volatile>(extents);
    STATIC_REQUIRE(std::is_same_v<typename ALPAKA_TYPEOF(hostBuffer)::value_type, T_DataType volatile>);
    auto hostBufferAsync = onHost::allocDeferred<T_DataType volatile>(onHost::makeHostDevice().makeQueue(), extents);
    STATIC_REQUIRE(std::is_same_v<typename ALPAKA_TYPEOF(hostBufferAsync)::value_type, T_DataType volatile>);
    auto hostBufferMapped = onHost::allocMapped<T_DataType volatile>(device, extents);
    STATIC_REQUIRE(std::is_same_v<typename ALPAKA_TYPEOF(hostBufferMapped)::value_type, T_DataType volatile>);
    auto deviceView = onHost::alloc<T_DataType volatile>(device, extents);
    STATIC_REQUIRE(std::is_same_v<typename ALPAKA_TYPEOF(deviceView)::value_type, T_DataType volatile>);
    auto deviceViewAsync = onHost::allocDeferred<T_DataType volatile>(device.makeQueue(), extents);
    STATIC_REQUIRE(std::is_same_v<typename ALPAKA_TYPEOF(deviceViewAsync)::value_type, T_DataType volatile>);
    auto unifiedView = onHost::allocUnified<T_DataType volatile>(device, extents);
    STATIC_REQUIRE(std::is_same_v<typename ALPAKA_TYPEOF(unifiedView)::value_type, T_DataType volatile>);
}

TEMPLATE_LIST_TEST_CASE("alloc volatile memory", "", TestDeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);
    INFO(deviceSpec.getApi().getName() << " on " << device.getName());

    using DataType = int;

    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (volatileBuffers<DataType>(device, extents), ...); }, extentMdList);
}
