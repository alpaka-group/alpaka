/* Copyright 2025 Anton Reinhard, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <type_traits>

using namespace alpaka;

using Data = std::size_t;
using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;
using DictWithoutDeviceSpecOrExec = Dict<DictEntry<object::Api, api::Host>>;
using DictWithoutExecutor = Dict<DictEntry<object::Api, api::Host>, DictEntry<object::DeviceKind, deviceKind::Cpu>>;

template<typename T_Idx>
void testVector()
{
    STATIC_CHECK(concepts::Vector<Vec<T_Idx, T_Idx{0}>>); // only disallowed at runtime
    STATIC_CHECK(concepts::Vector<Vec<Data, T_Idx{1}>>);
    STATIC_CHECK(concepts::Vector<Vec<Data, T_Idx{2}>>);
    STATIC_CHECK(concepts::Vector<Vec<Data, T_Idx{3}>>);
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<T_Idx, T_Idx{0}>>); // only disallowed at runtime
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<Data, T_Idx{1}>>);
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<Data, T_Idx{2}>>);
    STATIC_CHECK_FALSE(concepts::CVector<alpaka::Vec<Data, T_Idx{3}>>);

    STATIC_CHECK(concepts::Vector<alpaka::Vec<Data, T_Idx{1}>, Data>);
    STATIC_CHECK_FALSE(concepts::Vector<alpaka::Vec<double, T_Idx{1}>, float>);

    STATIC_CHECK(concepts::Vector<alpaka::CVec<Data, T_Idx{1}, Data{0}>>);
    STATIC_CHECK(concepts::Vector<alpaka::CVec<Data, T_Idx{2}, Data{0}, Data{0}>>);
    STATIC_CHECK(concepts::Vector<alpaka::CVec<Data, T_Idx{3}, Data{0}, Data{0}, Data{0}>>);
    STATIC_CHECK(concepts::CVector<alpaka::CVec<Data, T_Idx{1}, Data{0}>>);
    STATIC_CHECK(concepts::CVector<alpaka::CVec<Data, T_Idx{2}, Data{0}, Data{0}>>);
    STATIC_CHECK(concepts::CVector<alpaka::CVec<Data, T_Idx{3}, Data{0}, Data{0}, Data{0}>>);

    STATIC_CHECK_FALSE(concepts::Vector<Data[]>);
    STATIC_CHECK_FALSE(concepts::CVector<Data[]>);
    STATIC_CHECK_FALSE(concepts::Vector<Data[], Data>);
    STATIC_CHECK_FALSE(concepts::CVector<Data[], Data>);
}

TEST_CASE("vector concepts", "[concepts][vector]")
{
    testVector<std::uint16_t>();
    testVector<std::uint32_t>();
    testVector<std::uint64_t>();
    testVector<std::int16_t>();
    testVector<std::int32_t>();
    testVector<std::int64_t>();
}

TEST_CASE("frame spec concepts", "[concepts][framespec]")
{
    using namespace alpaka::onHost;

    using TestVec1 = Vec<size_t, 1>;
    using TestVec2 = Vec<size_t, 2>;
    using TestVec3 = Vec<size_t, 3>;

    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec1, TestVec1>>);
    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec2, TestVec2>, size_t>);
    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec3, TestVec3>, size_t, 3>);
    STATIC_CHECK(onHost::concepts::FrameSpec<FrameSpec<TestVec2, TestVec2>, size_t, 2>);

    STATIC_CHECK_FALSE(onHost::concepts::FrameSpec<FrameSpec<TestVec1, TestVec1>, int>);
    STATIC_CHECK_FALSE(onHost::concepts::FrameSpec<FrameSpec<TestVec2, TestVec2>, size_t, 3>);
    STATIC_CHECK_FALSE(onHost::concepts::FrameSpec<FrameSpec<TestVec3, TestVec3>, int, 2>);
}

TEST_CASE("thread spec concepts", "[concepts][threadspec]")
{
    using namespace alpaka::onHost;

    using TestVec1 = Vec<size_t, 1>;
    using TestVec2 = Vec<size_t, 2>;

    STATIC_CHECK(onHost::concepts::ThreadSpec<ThreadSpec<TestVec1, TestVec1>>);
    STATIC_CHECK(onHost::concepts::ThreadSpec<ThreadSpec<TestVec2, TestVec2>, size_t>);
    STATIC_CHECK(onHost::concepts::ThreadSpec<ThreadSpec<TestVec1, TestVec1>, size_t, 1>);

    STATIC_CHECK_FALSE(onHost::concepts::ThreadSpec<ThreadSpec<TestVec2, TestVec2>, int>);
    STATIC_CHECK_FALSE(onHost::concepts::ThreadSpec<ThreadSpec<TestVec1, TestVec1>, size_t, 2>);
}

TEMPLATE_LIST_TEST_CASE("backend concept", "[concepts][backend]", TestBackends)
{
    auto backend = TestType::makeDict();
    auto deviceSpec = onHost::DeviceSpec{backend};

    STATIC_CHECK(alpaka::concepts::BackendSpec<TestType>);
    STATIC_CHECK_FALSE(alpaka::concepts::BackendSpec<DictWithoutDeviceSpecOrExec>);
    STATIC_CHECK_FALSE(alpaka::concepts::BackendSpec<DictWithoutExecutor>);
    STATIC_CHECK(alpaka::concepts::DeviceSpec<decltype(deviceSpec)>);
    CHECK(alpaka::getApi(backend) == alpaka::getApi(deviceSpec));
    CHECK(alpaka::getDeviceKind(backend) == alpaka::getDeviceKind(deviceSpec));
}
