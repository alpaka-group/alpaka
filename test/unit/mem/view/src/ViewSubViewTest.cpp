/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>      // alpaka::test::acc::TestAccs

#include <boost/test/unit_test.hpp>

#include <type_traits>                  // std::is_same

BOOST_AUTO_TEST_SUITE(acc)


//#############################################################################
//!
//#############################################################################
struct CheckPitchBytesIdentical
{
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    CheckPitchBytesIdentical()
    {};

    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    template<
        typename TIdx,
        typename TView1,
        typename TView2>
    auto operator()(
        TView1 const & view1,
        TView2 const & view2) const
    -> void
    {
        BOOST_REQUIRE_EQUAL(
            alpaka::mem::view::getPitchBytes<TIdx::value>(view1),
            alpaka::mem::view::getPitchBytes<TIdx::value>(view2));
    }
};

//#############################################################################
//!
//#############################################################################
struct CheckPitchBytesIdentical2
{
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    CheckPitchBytesIdentical2()
    {};

    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    template<
        typename TIdx,
        typename TDim,
        typename TSize,
        typename TView2>
    auto operator()(
        alpaka::Vec<TDim, TSize> const & vec,
        TView2 const & view2) const
    -> void
    {
        BOOST_REQUIRE_EQUAL(
            vec[TIdx::value],
            alpaka::mem::view::getPitchBytes<TIdx::value>(view2));
    }
};

//#############################################################################
//! 1D: sizeof(TSize) * (5)
//! 2D: sizeof(TSize) * (5, 4)
//! 3D: sizeof(TSize) * (5, 4, 3)
//! 4D: sizeof(TSize) * (5, 4, 3, 2)
//#############################################################################
template<
    std::size_t Tidx>
struct CreateExtentBufVal
{
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    template<
        typename TSize>
    static auto create(
        TSize)
    -> TSize
    {
        return sizeof(TSize) * (5u - Tidx);
    }
};

//#############################################################################
//! 1D: sizeof(TSize) * (4)
//! 2D: sizeof(TSize) * (4, 3)
//! 3D: sizeof(TSize) * (4, 3, 2)
//! 4D: sizeof(TSize) * (4, 3, 2, 1)
//#############################################################################
template<
    std::size_t Tidx>
struct CreateExtentViewVal
{
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    template<
        typename TSize>
    static auto create(
        TSize)
    -> TSize
    {
        return sizeof(TSize) * (4u - Tidx);
    }
};

//-----------------------------------------------------------------------------
//!
//-----------------------------------------------------------------------------
template<
    typename TDim,
    typename TSize,
    template<std::size_t> class TCreate>
static auto createVecFromIndexedFn()
-> alpaka::Vec<TDim, TSize>
{
    return
#ifdef ALPAKA_CREATE_VEC_IN_CLASS
        alpaka::Vec<TDim, TSize>::template
#else
        alpaka::
#endif
        createVecFromIndexedFn<
#ifndef ALPAKA_CREATE_VEC_IN_CLASS
            TDim,
#endif
            TCreate>(
                TSize());
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    basicViewSubViewOperations,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    using View = alpaka::mem::view::ViewSubView<Dev, Elem, Dim, Size>;

    Dev dev(alpaka::dev::DevMan<TAcc>::getDevByIdx(0u));

    // We have to be careful with the extents used.
    // When Size is a 8 bit signed integer and Dim is 4, the extent is extremely limited.
    auto const extentBuf(createVecFromIndexedFn<Dim, Size, CreateExtentBufVal>());
    auto buf(alpaka::mem::buf::alloc<Elem, Size>(dev, extentBuf));

    // TODO: Test failing cases of view extents larger then the underlying buffer extents.
    auto const extentView(createVecFromIndexedFn<Dim, Size, CreateExtentViewVal>());
    auto const offsetView(alpaka::Vec<Dim, Size>::all(sizeof(Size)));
    View view(buf, extentView, offsetView);

    //-----------------------------------------------------------------------------
    // alpaka::dev::traits::DevType
    static_assert(
        std::is_same<alpaka::dev::Dev<View>, Dev>::value,
        "The device type of the view has to be equal to the specified one.");
    //-----------------------------------------------------------------------------
    // alpaka::dev::traits::GetDev
    BOOST_REQUIRE(
        dev == alpaka::dev::getDev(view));

    //-----------------------------------------------------------------------------
    // alpaka::dim::traits::DimType
    static_assert(
        alpaka::dim::Dim<View>::value == Dim::value,
        "The dimensionality of the view has to be equal to the specified one.");

    //-----------------------------------------------------------------------------
    // alpaka::elem::traits::ElemType
    static_assert(
        std::is_same<alpaka::elem::Elem<View>, Elem>::value,
        "The element type of the view has to be equal to the specified one.");

    //-----------------------------------------------------------------------------
    // alpaka::extent::traits::GetExtent
    BOOST_REQUIRE_EQUAL(
        extentView,
        alpaka::extent::getExtentVec(view));

    //-----------------------------------------------------------------------------
    // alpaka::mem::view::traits::GetPitchBytes
    // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
    using IdxSequence1 = alpaka::meta::MakeIndexSequence<Dim::value + 1u>;
    using DimSequence1 = alpaka::meta::TransformIntegerSequence<std::tuple, std::size_t, alpaka::dim::DimInt, IdxSequence1>;
    CheckPitchBytesIdentical const checkPitchBytesIdentical;
    alpaka::meta::forEachType<
        DimSequence1>(
            checkPitchBytesIdentical,
            buf,
            view);

    // The pitches have to be exactly the values we calculate here.
    auto pitches(alpaka::Vec<alpaka::dim::DimInt<Dim::value + 1u>, Size>::ones());
    // Initialize the pitch between two elements of the X dimension ...
    pitches[Dim::value] = sizeof(Elem);
    // ... and fill all the other dimensions.
    for(Size i = Dim::value; i > static_cast<Size>(0u); --i)
    {
        pitches[i-1] = extentBuf[i-1] * pitches[i];
    }
    CheckPitchBytesIdentical2 const checkPitchBytesIdentical2;
    alpaka::meta::forEachType<
        DimSequence1>(
            checkPitchBytesIdentical2,
            pitches,
            view);

    //-----------------------------------------------------------------------------
    // alpaka::mem::view::traits::GetPtrNative
    // The native pointer has to be exactly the value we calculate here.
    auto viewPtrNative(reinterpret_cast<std::uint8_t *>(alpaka::mem::view::getPtrNative(buf)));
    for(Size i = Dim::value; i > static_cast<Size>(0u); --i)
    {
        viewPtrNative += offsetView[i - 1u] * pitches[i];
    }
    BOOST_REQUIRE_EQUAL(
        reinterpret_cast<Elem *>(viewPtrNative),
        alpaka::mem::view::getPtrNative(view));

    //-----------------------------------------------------------------------------
    // alpaka::offset::traits::GetOffset
    BOOST_REQUIRE_EQUAL(
        offsetView,
        alpaka::offset::getOffsetVec(view));

    //-----------------------------------------------------------------------------
    // alpaka::size::traits::SizeType
    static_assert(
        std::is_same<alpaka::size::Size<View>, Size>::value,
        "The size type of the view has to be equal to the specified one.");
}

BOOST_AUTO_TEST_SUITE_END()
