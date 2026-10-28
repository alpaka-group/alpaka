/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

TEST_CASE(
    "InnerTypeAllowedCast"
    "[mem][concepts]")
{
    SECTION("GetElementType")
    {
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int&>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const&>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int[2][2]>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const[2][2]>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int (&)[2][2]>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const(&)[2][2]>::type, int>);

        STATIC_REQUIRE_FALSE(internal::GetElementType<int>::isConst);
        STATIC_REQUIRE(internal::GetElementType<int const>::isConst);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int&>::isConst);
        STATIC_REQUIRE(internal::GetElementType<int const&>::isConst);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int[2][2]>::isConst);
        STATIC_REQUIRE(internal::GetElementType<int const[2][2]>::isConst);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int (&)[2][2]>::isConst);
        STATIC_REQUIRE(internal::GetElementType<int const(&)[2][2]>::isConst);
    }

    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int, int>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const, int const>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const, int>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int, int const>);

    // check reference if type is working
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const&, int&>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const&, int>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const, int&>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int&, int const&>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int&, int const>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int, int const&>);

    // test C static array
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int[2][2], int[2][2]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const[2][2], int const[2][2]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const[2][2], int[2][2]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int[2][2], int const[2][2]>);

    // check if a reference to a C static array is working
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const(&)[1][1][1], int (&)[1][1][1]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const(&)[1][1][1], int[1][1][1]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const[1][1][1], int (&)[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int (&)[1][1][1], int const(&)[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int (&)[1][1][1], int const[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int[1][1][1], int const(&)[1][1][1]>);
}

// const value and const reference versions of the functions are not necessary
// the const modifier is not part of the deduction and therefore it will be not passed to the concept
void iMdSpanCallByValue(concepts::IMdSpan auto)
{
}

void iMdSpanCallByReference(concepts::IMdSpan auto&)
{
}

void iMdSpanCallByUniversalReference(concepts::IMdSpan auto&&)
{
}

/* TODO: remove me, if ICPX 2025.1 is not supported anymore.
 *
 * CATCH 3.15 use a Clang specific pragma to suppress the warning "-Wvariadic-macro-arguments-omitted".
 * The warning was introduced in Clang 20.1. ICPX 2025.1 based on Clang 20.0 (dev version).
 * Therefore, the detection if the warning is available does not work correctly, and we need to disable the warning
 * of unknown warnings temporary.
 */
#if ALPAKA_COMP_ICPX && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2025, 2, 0)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunknown-warning-option"
#endif

TEMPLATE_TEST_CASE_SIG(
    "IDataSource, IMdSpan, IView and IBuffer concept test",
    "[mem][concepts]",
    ((typename T_Elem, uint32_t T_dim), T_Elem, T_dim),
    (int, 1),
    (int, 2),
    (int, 3),
    (int const, 1),
    (int const, 2),
    (int const, 3))
{
    constexpr size_t size = 10;
    T_Elem* ptr = nullptr;
    concepts::Vector auto const extents = Vec<uint32_t, T_dim>{}.fill(size);
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    SECTION("test alpaka::LinearizedIdxGenerator object")
    {
        using GenType = alpaka::LinearizedIdxGenerator<uint32_t, T_dim>;
        [[maybe_unused]] GenType gen(extents);
        STATIC_REQUIRE(std::same_as<typename GenType::value_type, uint32_t>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<GenType>);
        STATIC_REQUIRE(alpaka::concepts::IDataSource<GenType const>);
        STATIC_REQUIRE(alpaka::concepts::IDataSource<GenType&>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<GenType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IView<GenType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<GenType>);
    }

    SECTION("test alpaka::MdSpan object")
    {
        MdSpan mdSpan(ptr, extents, pitches);
        using MdSpanType = decltype(mdSpan);
        STATIC_REQUIRE(std::same_as<typename MdSpanType::value_type, T_Elem>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<MdSpanType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType const>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType&>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IView<MdSpanType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<MdSpanType>);

        iMdSpanCallByValue(mdSpan);
        iMdSpanCallByReference(mdSpan);
        iMdSpanCallByUniversalReference(mdSpan);

        MdSpan const constMdSpan(ptr, extents, pitches);
        iMdSpanCallByValue(constMdSpan);
        iMdSpanCallByReference(constMdSpan);
        iMdSpanCallByUniversalReference(constMdSpan);

        iMdSpanCallByUniversalReference(MdSpan{ptr, extents, pitches});
    }

    SECTION("test alpaka::MdSpanArrayType object")
    {
        auto zero = static_cast<T_Elem>(0);
        T_Elem staticData[2][2] = {{zero, zero}, {zero, zero}};
        using MdSpanArrayType = MdSpanArray<decltype(staticData), size_t, alpaka::Alignment<16>>;
        MdSpanArrayType mdSpanArray(staticData);
        STATIC_REQUIRE(std::same_as<typename MdSpanArrayType::value_type, T_Elem>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<MdSpanArrayType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanArrayType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanArrayType const>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanArrayType&>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IView<MdSpanArrayType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<MdSpanArrayType>);

        iMdSpanCallByValue(mdSpanArray);
        iMdSpanCallByReference(mdSpanArray);
        iMdSpanCallByUniversalReference(mdSpanArray);

        MdSpanArrayType const constMdSpanArray(staticData);
        iMdSpanCallByValue(constMdSpanArray);
        iMdSpanCallByReference(constMdSpanArray);
        iMdSpanCallByUniversalReference(constMdSpanArray);

        iMdSpanCallByUniversalReference(MdSpanArrayType{staticData});
    }

    SECTION("test alpaka::View object")
    {
        View view(api::host, ptr, extents, pitches);
        using ViewType = decltype(view);
        STATIC_REQUIRE(std::same_as<typename ViewType::value_type, T_Elem>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<ViewType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<ViewType>);
        STATIC_REQUIRE(alpaka::concepts::IView<ViewType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<ViewType>);
    }

    SECTION("test alpaka::SharedBuffer object")
    {
        auto nullptrDeleter = [] {};
        onHost::SharedBuffer buffer(api::host, ptr, extents, pitches, nullptrDeleter);
        using BufferType = decltype(buffer);
        STATIC_REQUIRE(std::same_as<typename BufferType::value_type, T_Elem>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<BufferType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<BufferType>);
        STATIC_REQUIRE(alpaka::concepts::IView<BufferType>);
        STATIC_REQUIRE(alpaka::concepts::IBuffer<BufferType>);
    }
}

#if ALPAKA_COMP_ICPX && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2025, 2, 0)
#    pragma clang diagnostic pop
#endif


TEST_CASE("test alpaka::concepts::IMdSpan optional element type", "[mem][concepts]")
{
    constexpr size_t size = 10;
    float* ptr = nullptr;
    float const* constPtr = nullptr;
    constexpr concepts::Vector auto extents = Vec(size);
    constexpr concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    MdSpan mdSpan(ptr, extents, pitches);
    using MdSpanType = decltype(mdSpan);
    MdSpan innerConstMdSpan(constPtr, extents, pitches);
    using InnerConstMdSpanType = decltype(innerConstMdSpan);

    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<InnerConstMdSpanType>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType, float>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<InnerConstMdSpanType, float const>);
    STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<MdSpanType, int>);
    STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<MdSpanType, float const>);
    STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<InnerConstMdSpanType, float>);
}

struct NonCopyStruct
{
    NonCopyStruct() = default;
    NonCopyStruct(NonCopyStruct const&) = delete;
};

TEST_CASE(
    "test return value type of the access operator for alpaka::concepts::IDataSource and IMdSpan",
    "[mem][concept]")
{
    // This regression test models element types such as std::atomic<int>:
    // access must still work by reference even when the element type is not copyable.
    // The pitch vector has to be computed from the same element type used by MdSpan.
    NonCopyStruct mcs;

    STATIC_REQUIRE_FALSE(std::convertible_to<NonCopyStruct, NonCopyStruct>);
    STATIC_REQUIRE(std::convertible_to<NonCopyStruct&, NonCopyStruct&>);

    concepts::Vector auto const extents = Vec{1u};
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<NonCopyStruct>(extents);

    alpaka::MdSpan mdspan(&mcs, extents, pitches);

    using MdSpanNonCopyStruct = decltype(mdspan);

    STATIC_REQUIRE(alpaka::concepts::IDataSource<MdSpanNonCopyStruct>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanNonCopyStruct>);
}

template<typename T_MdSpan, typename T_Idx>
concept CanCallAccessOperator = requires(T_MdSpan mdspan, T_Idx idx) { mdspan[idx]; };

TEST_CASE("test alpaka::MdSpan argument concept access operator - correct dimension", "[mem][concepts][mdspan]")
{
    int data = 4;

    using Vec1D = alpaka::Vec<int, 1>;
    using Vec2D = alpaka::Vec<int, 2>;
    using Vec3D = alpaka::Vec<int, 3>;

    Vec1D extents1d{1};
    Vec2D extents2d{1, 1};
    Vec3D extents3d{1, 1, 1};

    Vec1D pitch1d = alpaka::calculatePitchesFromExtents<int>(extents1d);
    Vec2D pitch2d = alpaka::calculatePitchesFromExtents<int>(extents2d);
    Vec3D pitch3d = alpaka::calculatePitchesFromExtents<int>(extents3d);

    using MdSpan1D = alpaka::MdSpan<int, Vec1D, Vec1D>;
    using MdSpan2D = alpaka::MdSpan<int, Vec2D, Vec2D>;
    using MdSpan3D = alpaka::MdSpan<int, Vec3D, Vec3D>;


    MdSpan1D mdspan1d{&data, extents1d, pitch1d};
    MdSpan2D mdspan2d{&data, extents2d, pitch2d};
    MdSpan3D mdspan3d{&data, extents3d, pitch3d};

    Vec1D idx1D{0};
    Vec2D idx2D{0, 0};
    Vec3D idx3D{0, 0, 0};

    STATIC_REQUIRE(std::is_same_v<Vec1D::value_type, int>);

    SECTION("1D mdspan")
    {
        REQUIRE(mdspan1d[idx1D] == 4);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1D, Vec1D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1D, Vec2D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1D, Vec3D>);
    }

    SECTION("2D mdspan")
    {
        REQUIRE(mdspan2d[idx2D] == 4);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan2D, Vec1D>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan2D, Vec2D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan2D, Vec3D>);
    }

    SECTION("3D mdspan")
    {
        REQUIRE(mdspan3d[idx3D] == 4);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan3D, Vec1D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan3D, Vec2D>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan3D, Vec3D>);
    }
}

template<typename T_IndexType>
using Vec1DIdx = alpaka::Vec<T_IndexType, 1>;

template<typename T_IndexType>
using MdSpan1DIdx = alpaka::MdSpan<int, Vec1DIdx<T_IndexType>, Vec1DIdx<T_IndexType>>;

TEST_CASE("test alpaka::MdSpan argument concept access operator - correct type", "[mem][concepts][mdspan]")
{
    SECTION("same sign")
    {
        STATIC_REQUIRE(alpaka::concepts::IndexVec<Vec1DIdx<int32_t>, int32_t, 1>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<int32_t>>);

        STATIC_REQUIRE(alpaka::concepts::IndexVec<Vec1DIdx<int16_t>, int32_t, 1>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<int16_t>>);

        STATIC_REQUIRE_FALSE(alpaka::concepts::IndexVec<Vec1DIdx<int32_t>, int16_t, 1>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1DIdx<int16_t>, Vec1DIdx<int32_t>>);
    }

    SECTION("different sign")
    {
        STATIC_REQUIRE_FALSE(alpaka::concepts::IndexVec<Vec1DIdx<uint32_t>, int32_t, 1>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<uint32_t>>);

        STATIC_REQUIRE_FALSE(alpaka::concepts::IndexVec<Vec1DIdx<int32_t>, uint32_t, 1>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1DIdx<uint32_t>, Vec1DIdx<int32_t>>);

        STATIC_REQUIRE(alpaka::concepts::IndexVec<Vec1DIdx<uint16_t>, int32_t, 1>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<uint16_t>>);
    }
}
