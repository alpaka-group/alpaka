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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>              // alpaka::test::acc::TestAccs
#include <alpaka/test/stream/Stream.hpp>        // DefaultStream
#include <alpaka/test/mem/view/Iterator.hpp>    // Iterator

#include <boost/test/unit_test.hpp>

#include <type_traits>                          // std::is_same

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
//! Compares iterators element-wise
//-----------------------------------------------------------------------------
struct CompareBufferKernel {
    template<
        typename TAcc,
        typename TIterA,
        typename TIterB>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        TIterA beginA,
        TIterA endA,
        TIterB beginB) const
    {
        (void)acc;
        for(; beginA != endA; ++beginA, ++beginB)
        {
            BOOST_REQUIRE_EQUAL(
                *beginA,
                *beginB);
        }
    }
};

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

BOOST_AUTO_TEST_CASE_TEMPLATE(
    copyViewSubViewStatic,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using DevAcc = alpaka::dev::Dev<TAcc>;
    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using Stream = alpaka::test::stream::DefaultStream<DevAcc>;

    using View = alpaka::mem::view::ViewSubView<DevAcc, Elem, Dim, Size>;
    using ViewPlainPtr = alpaka::mem::view::ViewPlainPtr<DevHost, Elem, Dim, Size>;

    const Size nElementsPerDim = static_cast<Size>(4u);
    const Size nElementsPerDimView = static_cast<Size>(2u);
    const Size offsetInAllDims = static_cast<Size>(1u);

    DevHost devHost(alpaka::dev::DevMan<Host>::getDevByIdx(0u));
    DevAcc devAcc(alpaka::dev::DevMan<TAcc>::getDevByIdx(0u));
    Stream stream(devAcc);

    using Vec = alpaka::Vec<Dim, Size>;

    const auto elementsPerThread(Vec::all(static_cast<Size>(1u)));
    const auto threadsPerBlock(Vec::all(static_cast<Size>(1u)));
    const auto blocksPerGrid(Vec::all(static_cast<Size>(1u)));

    WorkDiv const workdiv(
        alpaka::workdiv::WorkDivMembers<Dim, Size>(
            blocksPerGrid,
            threadsPerBlock,
            elementsPerThread));

    auto const extentBuf(Vec::all(nElementsPerDim));
    auto const extentView(Vec::all(nElementsPerDimView));
    auto const offsetView(Vec::all(offsetInAllDims));
    auto buf(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extentBuf));
    auto buf2(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extentView));
    View view(buf, extentView, offsetView);

    // Init buf with increasing values
    std::vector<Elem> v(extentBuf.prod(), static_cast<Elem>(0u));
    std::iota(v.begin(), v.end(), static_cast<Elem>(0u));
    ViewPlainPtr plainBuf(v.data(), devHost, extentBuf);
    alpaka::mem::view::copy(stream, buf, plainBuf, extentBuf);

    CompareBufferKernel compareBufferKernel;

    switch(Dim::value)
    {
    case 1:
        {
            std::vector<Elem> v2{1,2};
            ViewPlainPtr plainBuf2(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, buf2, plainBuf2, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(buf2),
                    alpaka::test::mem::view::end(buf2),
                    alpaka::test::mem::view::begin(view)));
            alpaka::stream::enqueue(stream, compare);
            alpaka::wait::wait(stream);
        break;
        }
    case 2:
        {
            std::vector<Elem> v2{5, 6, 9, 10};
            ViewPlainPtr plainBuf2(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, buf2, plainBuf2, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(buf2),
                    alpaka::test::mem::view::end(buf2),
                    alpaka::test::mem::view::begin(view)));
            alpaka::stream::enqueue(stream, compare);
            alpaka::wait::wait(stream);
            break;
        }
    case 3:
        {
            std::vector<Elem> v2{21, 22, 25, 26, 37, 38, 41, 42};
            ViewPlainPtr plainBuf2(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, buf2, plainBuf2, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(buf2),
                    alpaka::test::mem::view::end(buf2),
                    alpaka::test::mem::view::begin(view)));
            alpaka::stream::enqueue(stream, compare);
            alpaka::wait::wait(stream);
            break;
        }
    case 4:
        {
            /*
            std::vector<Elem> v2{75, 76, 78, 79, 91, 92, 95, 96, 139, 140, 143, 144, 155, 156, 159, 160};
            ViewPlainPtr plainBuf2(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, buf2, plainBuf2, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(buf2),
                    alpaka::test::mem::view::end(buf2),
                    alpaka::test::mem::view::begin(view)));
            */
            alpaka::wait::wait(stream);
            break;
        }
    default:
        alpaka::wait::wait(stream);
        break;
    };
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    copyViewSubViewGeneric,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using DevAcc = alpaka::dev::Dev<TAcc>;
    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using Stream = alpaka::test::stream::DefaultStream<DevAcc>;

    using View = alpaka::mem::view::ViewSubView<DevAcc, Elem, Dim, Size>;
    using ViewPlainPtr = alpaka::mem::view::ViewPlainPtr<DevHost, Elem, Dim, Size>;

    if(Dim::value != 4)
    {
        DevHost devHost (alpaka::dev::DevMan<Host>::getDevByIdx(0));
        DevAcc devAcc(alpaka::dev::DevMan<TAcc>::getDevByIdx(0u));
        Stream stream (devAcc);

        auto const extentBuf(createVecFromIndexedFn<Dim, Size, CreateExtentBufVal>());
        auto const extentView(createVecFromIndexedFn<Dim, Size, CreateExtentViewVal>());
        auto const offsetView(alpaka::Vec<Dim, Size>::all(sizeof(Size)));
        auto buf(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extentBuf));
        auto buf2(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extentView));
        View view(buf, extentView, offsetView);

        //-----------------------------------------------------------------------------
        // alpaka::mem::view::copy
        // Init buf with increasing values
        std::vector<Elem> v(extentBuf.prod(), static_cast<Elem>(0));
        std::iota(v.begin(), v.end(), static_cast<Elem>(0));
        ViewPlainPtr plainBuf(v.data(), devHost, extentBuf);
        alpaka::mem::view::copy(stream, buf, plainBuf, extentBuf);

        // Copy view into 2nd buf
        alpaka::mem::view::copy(stream, buf2, view, extentView);

        // Check values in 2nd buf
        alpaka::Vec<Dim, Size> elementsPerThread(alpaka::Vec<Dim, Size>::all(static_cast<Size>(1)));
        alpaka::Vec<Dim, Size> threadsPerBlock(alpaka::Vec<Dim, Size>::all(static_cast<Size>(1)));
        alpaka::Vec<Dim, Size> const blocksPerGrid(alpaka::Vec<Dim, Size>::all(static_cast<Size>(1)));

        WorkDiv const workdiv(
            alpaka::workdiv::WorkDivMembers<Dim, Size>(
                blocksPerGrid,
                threadsPerBlock,
                elementsPerThread));

        CompareBufferKernel compareBufferKernel;
        auto const compare(
            alpaka::exec::create<TAcc>(
                workdiv,
                compareBufferKernel,
                alpaka::test::mem::view::begin(buf2),
                alpaka::test::mem::view::end(buf2),
                alpaka::test::mem::view::begin(view)));

        alpaka::stream::enqueue(stream, compare);
        alpaka::wait::wait(stream);
    }
}

BOOST_AUTO_TEST_SUITE_END()
