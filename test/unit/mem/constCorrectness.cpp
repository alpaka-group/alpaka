/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/mem/concepts/detail/CopyConstructableDataSource.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <iostream>

using namespace alpaka;

/** @file test the behaviour for constness of buffers/views/MdSpan
 *
 * We run two different test, one for access with 1D memory and the usage of scalars to access the memory and one for
 * multidimensional memory, because there are specializations for 1D in our implementations.
 */

/** validate that the buffer is mutable
 *
 * If we use non-outer const buffers/views in the function signature it should be possible to call the function even if
 * the buffer/view had an outer const before. The reason why we can not transform the outer const into an inner const
 * is that we requires this within @c KernelBundle to forward the kernel user arguments in a tuple to the user kernel.
 */
void requiresMutableBufferMd(concepts::IMdSpan auto buffer)
{
    STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(buffer.data())>>);
    STATIC_REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(buffer[Vec{0, 0}])>>);
}

TEST_CASE("mdspan inner const copy constructor", "[mem][mdspan][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutMdSpan>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutMdSpan>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstMdSpan>);

    MutMdSpan mutMdspan(ptr, extents, pitches);
    ConstMdSpan constMdspan(constPtr, extents, pitches);

    STATIC_REQUIRE(std::constructible_from<MutMdSpan, MutMdSpan&>);
    [[maybe_unused]] MutMdSpan mutMdspanCopy(mutMdspan);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpan, ConstMdSpan&>);
    [[maybe_unused]] ConstMdSpan constMdspanCopy(constMdspan);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpan, MutMdSpan&>);
    [[maybe_unused]] ConstMdSpan mutToConstMdspan(mutMdspan);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutMdSpan, ConstMdSpan&>);
    // should not compile
    // MdSpan const_to_mud_mdspan(const_mdspan);
}

TEST_CASE("mdspan inner const assignment operator", "[mem][mdspan][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    MutMdSpan mutMdspan(ptr, extents, pitches);
    ConstMdSpan constMdspan(constPtr, extents, pitches);

    STATIC_REQUIRE(concepts::AssignableFrom<MutMdSpan&, MutMdSpan>);
    [[maybe_unused]] MutMdSpan mutMdspan2 = mutMdspan;
    mutMdspan2 = mutMdspan;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpan&, ConstMdSpan>);
    [[maybe_unused]] ConstMdSpan constMdspan2 = constMdspan;
    constMdspan2 = constMdspan;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpan&, MutMdSpan>);
    [[maybe_unused]] ConstMdSpan constMdspan3 = mutMdspan;
    constMdspan3 = mutMdspan;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutMdSpan&, ConstMdSpan>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutMdSpan mut_mdspan3 = const_mdspan;
    // mut_mdspan3 = const_mdspan;
}

TEST_CASE("mdspan inner const move operator", "[mem][mdspan][correctness][moveConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    MutMdSpan copyMutMdspan(ptr, extents, pitches);
    ConstMdSpan copyConstMdspan(constPtr, extents, pitches);

    [[maybe_unused]] MutMdSpan copyMutMdspan2(std::move(copyMutMdspan));
    [[maybe_unused]] ConstMdSpan copyConstMdspan2(std::move(copyConstMdspan));
    [[maybe_unused]] ConstMdSpan copyConstMdspan3(std::move(copyMutMdspan2));
    // should not compile
    // MutMdSpan copy_mut_mdspan3(std::move(copy_const_mdspan3));

    // at the moment it is more performant efficient, that a mdspan is still valid after a move
    CHECK(copyMutMdspan);
    CHECK(copyConstMdspan);
    CHECK(copyMutMdspan2);
    CHECK(copyConstMdspan2);
    CHECK(copyConstMdspan3);

    MutMdSpan assignMutMdspan(ptr, extents, pitches);
    ConstMdSpan assignConstMdspan(constPtr, extents, pitches);

    MutMdSpan assignMutMdspan2 = std::move(assignMutMdspan);
    [[maybe_unused]] ConstMdSpan assignConstMdspan2 = std::move(assignConstMdspan);
    [[maybe_unused]] ConstMdSpan assignConstMdspan3 = std::move(assignMutMdspan2);
    // should not compile
    // MutMdSpan assign_mut_mdspan3 = std::move(assign_const_mdspan3);

    // at the moment it is more performant efficient, that a mdspan is still valid after a move
    CHECK(assignMutMdspan);
    CHECK(assignConstMdspan);
    CHECK(assignMutMdspan2);
    CHECK(assignConstMdspan2);
    CHECK(assignConstMdspan3);
}

template<typename T_ExpectedValueType, typename T_ExpectedReferenceType, typename T_MdSpan>
requires internal::concepts::CopyConstructableDataSource<T_MdSpan> && concepts::IMdSpan<T_MdSpan>
void funcCopyByValue(T_MdSpan mdspan)
{
    STATIC_REQUIRE(std::same_as<typename decltype(mdspan)::value_type, T_ExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), T_ExpectedReferenceType&>);
}

template<typename T_ExpectedValueType, typename T_ExpectedReferenceType, typename T_MdSpan>
requires internal::concepts::CopyConstructableDataSource<T_MdSpan> && concepts::IMdSpan<T_MdSpan>
void funcReference(T_MdSpan& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, T_ExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), T_ExpectedReferenceType&>);
}

template<typename T_ExpectedValueType, typename T_ExpectedReferenceType, typename T_MdSpan>
requires internal::concepts::CopyConstructableDataSource<T_MdSpan> && concepts::IMdSpan<T_MdSpan>
void funcConstReference(T_MdSpan const& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, T_ExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), T_ExpectedReferenceType&>);
}

template<typename T_ExpectedValueType, typename T_ExpectedReferenceType, typename T_MdSpan>
requires internal::concepts::CopyConstructableDataSource<T_MdSpan> && concepts::IMdSpan<T_MdSpan>
void funcUniversalRef(T_MdSpan&& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, T_ExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), T_ExpectedReferenceType&>);
}

TEST_CASE("function calls with mdspan object", "[mem][mdspan][correctness]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 1>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    MdSpan mdspan{ptr, extents, pitches};
    funcCopyByValue<int, int&>(mdspan);
    funcReference<int, int&>(mdspan);
    funcConstReference<int, int const&>(mdspan);
    funcUniversalRef<int, int&>(mdspan);
    funcUniversalRef<int, int&>(MdSpan{ptr, extents, pitches});

    MdSpan const constMdspan{ptr, extents, pitches};
    funcCopyByValue<int, int&>(constMdspan);
    funcReference<int, int const&>(constMdspan);
    funcConstReference<int, int const&>(constMdspan);

    MdSpan mdspanInnerConst{constPtr, extents, pitches};
    funcCopyByValue<int const, int const&>(mdspanInnerConst);
    funcReference<int const, int const&>(mdspanInnerConst);
    funcConstReference<int const, int const&>(mdspanInnerConst);
    funcUniversalRef<int const, int const&>(mdspanInnerConst);
    funcUniversalRef<int const, int const&>(MdSpan{constPtr, extents, pitches});

    MdSpan const constMdspanInnerConst{constPtr, extents, pitches};
    funcCopyByValue<int const, int const&>(constMdspanInnerConst);
    funcReference<int const, int const&>(constMdspanInnerConst);
    funcConstReference<int const, int const&>(constMdspanInnerConst);
    funcUniversalRef<int const, int const&>(constMdspanInnerConst);
}

TEST_CASE("MdSpanArray inner const copy constructor", "[mem][mdspanarray][correctness][copyConstruct]")
{
    int staticData[2][2] = {{0, 0}, {0, 0}};
    int const constStaticData[2][2] = {{0, 0}, {0, 0}};
    using MutMdSpanArray = MdSpanArray<decltype(staticData), size_t, alpaka::Alignment<16>>;
    using ConstMdSpanArray = MdSpanArray<decltype(constStaticData), size_t, alpaka::Alignment<16>>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutMdSpanArray>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutMdSpanArray>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstMdSpanArray>);

    MutMdSpanArray mutMdSpanArray(staticData);
    ConstMdSpanArray constMdSpanArray(constStaticData);

    STATIC_REQUIRE(std::constructible_from<MutMdSpanArray, MutMdSpanArray&>);
    [[maybe_unused]] MutMdSpanArray mutMdSpanArrayCopy(mutMdSpanArray);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpanArray, ConstMdSpanArray&>);
    [[maybe_unused]] ConstMdSpanArray constMdSpanArrayCopy(constMdSpanArray);


    STATIC_REQUIRE(std::constructible_from<ConstMdSpanArray, MutMdSpanArray&>);
    [[maybe_unused]] ConstMdSpanArray mutToConstMdSpanArray(mutMdSpanArray);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutMdSpanArray, ConstMdSpanArray&>);
    // should not compile
    // MutMdSpanArray const_to_mut_md_span_array(const_md_span_array);
}

TEST_CASE("MdSpanArray inner const assignment operator", "[mem][mdspanarray][correctness][copyConstruct]")
{
    int staticData[2][2] = {{0, 0}, {0, 0}};
    int const constStaticData[2][2] = {{0, 0}, {0, 0}};
    using MutMdSpanArray = MdSpanArray<decltype(staticData), size_t, alpaka::Alignment<16>>;
    using ConstMdSpanArray = MdSpanArray<decltype(constStaticData), size_t, alpaka::Alignment<16>>;

    MutMdSpanArray mutMdSpanArray(staticData);
    ConstMdSpanArray constMdSpanArray(constStaticData);

    STATIC_REQUIRE(concepts::AssignableFrom<MutMdSpanArray&, MutMdSpanArray>);
    [[maybe_unused]] MutMdSpanArray mutMdSpanArra2 = mutMdSpanArray;
    mutMdSpanArra2 = mutMdSpanArray;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpanArray&, ConstMdSpanArray>);
    [[maybe_unused]] ConstMdSpanArray constMdSpanArray2 = constMdSpanArray;
    constMdSpanArray2 = constMdSpanArray;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstMdSpanArray&, MutMdSpanArray>);
    [[maybe_unused]] ConstMdSpanArray constMdSpanArray3 = mutMdSpanArray;
    constMdSpanArray3 = mutMdSpanArray;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutMdSpanArray&, ConstMdSpanArray>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutMdSpanArray mut_view3 = const_md_span_array;
    // mut_view3 = const_md_span_array;
}

TEST_CASE(
    "MdSpanArray inner const move constructor and move assignment operator",
    "[mem][mdspanarray][correctness][moveConstruct]")
{
    int staticData[2][2] = {{0, 0}, {0, 0}};
    int const constStaticData[2][2] = {{0, 0}, {0, 0}};
    using MutMdSpanArray = MdSpanArray<decltype(staticData), size_t, alpaka::Alignment<16>>;
    using ConstMdSpanArray = MdSpanArray<decltype(constStaticData), size_t, alpaka::Alignment<16>>;

    MutMdSpanArray constrMutMd(staticData);
    ConstMdSpanArray constrConstMd(constStaticData);

    MutMdSpanArray constrMutMd2(std::move(constrMutMd));
    [[maybe_unused]] ConstMdSpanArray constrConstMd2(std::move(constrConstMd));
    [[maybe_unused]] ConstMdSpanArray constrConstMd3(std::move(constrMutMd2));
    // should not compile
    // MutMdSpanArray constr_mut_md3(std::move(constr_const_md3));

    // at the moment it is more performant efficient, that a mdspan is still valid after a move
    CHECK(constrMutMd);
    CHECK(constrConstMd);
    CHECK(constrMutMd2);
    CHECK(constrConstMd2);

    MutMdSpanArray assignMutMd(staticData);
    ConstMdSpanArray assignConstMd(constStaticData);

    MutMdSpanArray assignMutMd2 = std::move(assignMutMd);
    [[maybe_unused]] ConstMdSpanArray assignConstMd2 = std::move(assignConstMd);
    [[maybe_unused]] ConstMdSpanArray assignConstMd3 = std::move(assignMutMd2);
    // should not compile
    // MutMdSpanArray assign_mut_md3 = std::move(assign_const_md3);

    // at the moment it is more performant efficient, that a mdspan is still valid after a move
    CHECK(assignMutMd);
    CHECK(assignConstMd);
    CHECK(assignMutMd2);
    CHECK(assignConstMd2);
    CHECK(assignConstMd3);
}

TEST_CASE("View inner const copy constructor", "[mem][view][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutView>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutView>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstView>);

    MutView mutView(api::host, ptr, extents, pitches);
    ConstView constView(api::host, constPtr, extents, pitches);

    STATIC_REQUIRE(std::constructible_from<MutView, MutView&>);
    [[maybe_unused]] MutView mutViewCopy(mutView);

    STATIC_REQUIRE(std::constructible_from<ConstView, ConstView&>);
    [[maybe_unused]] ConstView constViewCopy(constView);

    STATIC_REQUIRE(std::constructible_from<ConstView, MutView&>);
    [[maybe_unused]] ConstView mutToConstView(mutView);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutView, ConstView&>);
    // should not compile
    // MutView const_to_mud_view(const_view);
}

TEST_CASE("View inner const assignment operator", "[mem][view][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    MutView mutView(api::host, ptr, extents, pitches);
    ConstView constView(api::host, constPtr, extents, pitches);

    STATIC_REQUIRE(concepts::AssignableFrom<MutView&, MutView>);
    [[maybe_unused]] MutView mutView2 = mutView;
    mutView2 = mutView;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstView&, ConstView>);
    [[maybe_unused]] ConstView constView2 = constView;
    constView2 = constView;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstView&, MutView>);
    [[maybe_unused]] ConstView constView3 = mutView;
    constView3 = mutView;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutView&, ConstView>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutView mut_view3 = const_view;
    // mut_view3 = const_view;
}

TEST_CASE("View inner const move constructor and move assignment operator", "[mem][view][correctness][moveConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    MutView constructMutView(api::host, ptr, extents, pitches);
    ConstView constructConstView(api::host, constPtr, extents, pitches);

    MutView constructMutView2(std::move(constructMutView));
    [[maybe_unused]] ConstView constructConstView2(std::move(constructConstView));
    [[maybe_unused]] ConstView constructConstView3(std::move(constructMutView2));
    // should not compile
    // MutView construct_mut_view3(std::move(construct_const_view3));

    // at the moment it is more performant efficient, that a view is still valid after a move
    CHECK(constructMutView);
    CHECK(constructConstView);
    CHECK(constructMutView2);
    CHECK(constructConstView2);
    CHECK(constructConstView3);

    MutView assignMutView(api::host, ptr, extents, pitches);
    ConstView assignConstView(api::host, constPtr, extents, pitches);

    MutView assignMutView2 = std::move(assignMutView);
    [[maybe_unused]] ConstView assignConstView2 = std::move(assignConstView);
    [[maybe_unused]] ConstView assignConstView3 = std::move(assignMutView2);
    // should not compile
    // MutView assign_mut_view3 = std::move(assign_const_view3);

    // at the moment it is more performant efficient, that a view is still valid after a move
    CHECK(assignMutView);
    CHECK(assignConstView);
    CHECK(assignMutView2);
    CHECK(assignConstView2);
    CHECK(assignConstView3);
}

TEST_CASE("sharedBuffer inner const copy constructor", "[mem][sharedBuffer][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int, decltype(extents)>;
    using ConstSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int const, decltype(extents)>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutSharedBuffer>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutSharedBuffer>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstSharedBuffer>);

    MutSharedBuffer mutSharedBuffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer constSharedBuffer(api::host, constPtr, extents, pitches, [] {});

    STATIC_REQUIRE(std::constructible_from<MutSharedBuffer, MutSharedBuffer&>);
    [[maybe_unused]] MutSharedBuffer mutSharedBufferCopy(mutSharedBuffer);

    STATIC_REQUIRE(std::constructible_from<ConstSharedBuffer, ConstSharedBuffer&>);
    [[maybe_unused]] ConstSharedBuffer constSharedBufferCopy(constSharedBuffer);

    STATIC_REQUIRE(std::constructible_from<ConstSharedBuffer, MutSharedBuffer&>);
    [[maybe_unused]] ConstSharedBuffer mutToConstSharedBuffer(mutSharedBuffer);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutSharedBuffer, ConstSharedBuffer&>);
    // should not compile
    // MutSharedBuffer const_to_mud_shared_buffer(const_shared_buffer);
}

TEST_CASE("sharedBuffer inner const assignment operator", "[mem][sharedBuffer][correctness][copyConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int, decltype(extents)>;
    using ConstSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int const, decltype(extents)>;

    MutSharedBuffer mutSharedBuffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer constSharedBuffer(api::host, constPtr, extents, pitches, [] {});

    STATIC_REQUIRE(concepts::AssignableFrom<MutSharedBuffer&, MutSharedBuffer>);
    [[maybe_unused]] MutSharedBuffer mutSharedBuffer2 = mutSharedBuffer;
    mutSharedBuffer2 = mutSharedBuffer;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstSharedBuffer&, ConstSharedBuffer>);
    [[maybe_unused]] ConstSharedBuffer constSharedBuffer2 = constSharedBuffer;
    constSharedBuffer2 = constSharedBuffer;
    STATIC_REQUIRE(concepts::AssignableFrom<ConstSharedBuffer&, MutSharedBuffer>);
    [[maybe_unused]] ConstSharedBuffer constSharedBuffer3 = mutSharedBuffer;
    constSharedBuffer3 = mutSharedBuffer;
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<MutSharedBuffer&, ConstSharedBuffer>);
    // Not allowed cases, remove inner const via copy construction or assignment
    // MutSharedBuffer mut_shared_buffer3 = const_shared_buffer;
    // mut_shared_buffer3 = const_shared_buffer;
}

TEST_CASE("sharedBuffer inner const move operator", "[mem][sharedBuffer][correctness][moveConstruct]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* constPtr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.fill(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int, decltype(extents)>;
    using ConstSharedBuffer = onHost::SharedBuffer<alpaka::api::Host, int const, decltype(extents)>;

    MutSharedBuffer copyMutSharedBuffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer copyConstSharedBuffer(api::host, constPtr, extents, pitches, [] {});

    MutSharedBuffer copyMutSharedBuffer2(std::move(copyMutSharedBuffer));
    ConstSharedBuffer copyConstSharedBuffer2(std::move(copyConstSharedBuffer));
    [[maybe_unused]] ConstSharedBuffer copyConstSharedBuffer3(std::move(copyMutSharedBuffer2));
    // should not compile
    // MutSharedBuffer copy_mut_shared_buffer3(std::move(copy_const_shared_buffer3));

    // check lifetime
    CHECK_FALSE(copyMutSharedBuffer);
    CHECK_FALSE(copyConstSharedBuffer);
    CHECK_FALSE(copyMutSharedBuffer2);
    CHECK(copyConstSharedBuffer2);
    CHECK(copyConstSharedBuffer3);

    MutSharedBuffer assignMutSharedBuffer(api::host, ptr, extents, pitches, [] {});
    ConstSharedBuffer assignConstSharedBuffer(api::host, constPtr, extents, pitches, [] {});

    MutSharedBuffer assignMutSharedBuffer2 = std::move(assignMutSharedBuffer);
    ConstSharedBuffer assignConstSharedBuffer2 = std::move(assignConstSharedBuffer);
    [[maybe_unused]] ConstSharedBuffer assignConstSharedBuffer3 = std::move(assignMutSharedBuffer2);
    // should not compile
    // MutSharedBuffer assign_mut_shared_buffer3 = std::move(assign_const_shared_buffer3);

    // check lifetime
    CHECK_FALSE(assignMutSharedBuffer);
    CHECK_FALSE(assignMutSharedBuffer2);
    CHECK_FALSE(assignConstSharedBuffer);
    CHECK(assignConstSharedBuffer2);
    CHECK(assignConstSharedBuffer3);
}

TEST_CASE("buffer const correctness MD", "[mem][sharedBuffer][correctness]")
{
    auto buffer0 = onHost::allocHost<int>(Vec{10, 10});
    requiresMutableBufferMd(buffer0);
    STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(buffer0.data())>>);

    SECTION("getSubView/getSubBuffer from mutable DataStorage object")
    {
        [[maybe_unused]] auto subBuffer0 = buffer0.getSubSharedBuffer(Vec{2, 2});
        STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        STATIC_REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = buffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        STATIC_REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = buffer0.getSubView(Vec{2, 2});
        STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        STATIC_REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = buffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        STATIC_REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = buffer0.getView();
        STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        STATIC_REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = buffer0.getMdSpan();
        STATIC_REQUIRE(!std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        STATIC_REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }

    SECTION("getSubView/getSubBuffer from non-mutable DataStorage object (inner const)")
    {
        onHost::SharedBuffer innerConstBuffer0 = buffer0.getConstSharedBuffer();

        [[maybe_unused]] auto subBuffer0 = innerConstBuffer0.getSubSharedBuffer(Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = innerConstBuffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = innerConstBuffer0.getSubView(Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = innerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = innerConstBuffer0.getView();
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = innerConstBuffer0.getMdSpan();
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }

    SECTION("getSubView/getSubBuffer from non-mutable DataStorage object (outer const)")
    {
        onHost::SharedBuffer const outerConstBuffer0 = buffer0;
        requiresMutableBufferMd(outerConstBuffer0);

        [[maybe_unused]] auto subBuffer0 = outerConstBuffer0.getSubSharedBuffer(Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = outerConstBuffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = outerConstBuffer0.getSubView(Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = outerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = outerConstBuffer0.getView();
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = outerConstBuffer0.getMdSpan();
        STATIC_REQUIRE(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }
}
