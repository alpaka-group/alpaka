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
#include <alpaka/test/mem/view/ViewTest.hpp>    // viewTest
#include <alpaka/test/mem/view/Iterator.hpp>    // Iterator

#include <boost/assert.hpp>                     // BOOST_VERIFY
#include <boost/predef.h>                       // BOOST_COMP_MSVC, BOOST_COMP_CLANG
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <type_traits>                          // std::is_same
#include <numeric>                              // std::iota

BOOST_AUTO_TEST_SUITE(memView)

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

//#############################################################################
//! Compares iterators element-wise
//#############################################################################
struct CompareBufferKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TIterA,
        typename TIterB>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        TIterA beginA,
        TIterA const & endA,
        TIterB beginB) const
    {
        (void)acc;
        for(; beginA != endA; ++beginA, ++beginB)
        {
            BOOST_VERIFY(*beginA == *beginB);
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
    using Pltf = alpaka::pltf::Pltf<Dev>;

    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;
    using View = alpaka::mem::view::ViewSubView<Dev, Elem, Dim, Size>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

    // We have to be careful with the extents used.
    // When Size is a 8 bit signed integer and Dim is 4, the extent is extremely limited.
    auto const extentBuf(createVecFromIndexedFn<Dim, Size, CreateExtentBufVal>());
    auto buf(alpaka::mem::buf::alloc<Elem, Size>(dev, extentBuf));

    // TODO: Test failing cases of view extents larger then the underlying buffer extents.
    auto const extentView(createVecFromIndexedFn<Dim, Size, CreateExtentViewVal>());
    auto const offsetView(alpaka::Vec<Dim, Size>::all(sizeof(Size)));
    View view(buf, extentView, offsetView);

    //-----------------------------------------------------------------------------
    viewTest<
        Elem>(
            view,
            dev,
            extentView,
            offsetView);

    //-----------------------------------------------------------------------------
    // alpaka::mem::view::traits::GetPitchBytes
    // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
    {
        auto const pitchBuf(alpaka::mem::view::getPitchBytesVec(buf));
        auto const pitchView(alpaka::mem::view::getPitchBytesVec(view));

        for(Size i = Dim::value; i > static_cast<Size>(0u); --i)
        {
            BOOST_REQUIRE_EQUAL(
                pitchBuf[i-1u],
                pitchView[i-1u]);
        }
    }

    //-----------------------------------------------------------------------------
    // alpaka::mem::view::traits::GetPtrNative
    // The native pointer has to be exactly the value we calculate here.
    {
        auto viewPtrNative(
            reinterpret_cast<std::uint8_t *>(
                alpaka::mem::view::getPtrNative(buf)));
        auto const pitchBuf(alpaka::mem::view::getPitchBytesVec(buf));
        for(Size i = Dim::value; i > static_cast<Size>(0u); --i)
        {
            auto const pitch = (i < static_cast<Size>(Dim::value)) ? pitchBuf[i] : static_cast<Size>(sizeof(Elem));
            viewPtrNative += offsetView[i - 1u] * pitch;
        }
        BOOST_REQUIRE_EQUAL(
            reinterpret_cast<Elem *>(viewPtrNative),
            alpaka::mem::view::getPtrNative(view));
    }
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    copyViewSubViewStatic,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;

    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;

    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;

    using ViewSubView = alpaka::mem::view::ViewSubView<DevAcc, Elem, Dim, Size>;
    using ViewPlainPtr = alpaka::mem::view::ViewPlainPtr<DevHost, Elem, Dim, Size>;

    const Size nElementsPerDim = static_cast<Size>(4u);
    const Size nElementsPerDimView = static_cast<Size>(2u);
    const Size offsetInAllDims = static_cast<Size>(1u);

    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    StreamAcc stream(devAcc);

    using Vec = alpaka::Vec<Dim, Size>;

    const auto elementsPerThread(Vec::ones());
    const auto threadsPerBlock(Vec::ones());
    const auto blocksPerGrid(Vec::ones());

    WorkDiv const workdiv(
        alpaka::workdiv::WorkDivMembers<Dim, Size>(
            blocksPerGrid,
            threadsPerBlock,
            elementsPerThread));

    // Init buf with increasing values
    auto const extentBuf(Vec::all(nElementsPerDim));
    auto bufAcc(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extentBuf));
    std::vector<Elem> v(static_cast<std::size_t>(extentBuf.prod()), static_cast<Elem>(0u));
    std::iota(v.begin(), v.end(), static_cast<Elem>(0u));
    ViewPlainPtr viewHost(v.data(), devHost, extentBuf);
    alpaka::mem::view::copy(stream, bufAcc, viewHost, extentBuf);

    // Create sub-view for buf.
    auto const extentView(Vec::all(nElementsPerDimView));
    auto const offsetView(Vec::all(offsetInAllDims));
    ViewSubView subViewAcc(bufAcc, extentView, offsetView);

    // Create 2nd buffer only containing the sub-view.
    auto referenceBufAcc(alpaka::mem::buf::alloc<Elem, Size>(devAcc, extentView));

    CompareBufferKernel compareBufferKernel;

    switch(Dim::value)
    {
    case 1:
        {
            std::vector<Elem> v2{1,2};
            ViewPlainPtr referenceViewHost(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, referenceBufAcc, referenceViewHost, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(referenceBufAcc),
                    alpaka::test::mem::view::end(referenceBufAcc),
                    alpaka::test::mem::view::begin(subViewAcc)));
            alpaka::stream::enqueue(stream, compare);
            alpaka::wait::wait(stream);
        break;
        }
    case 2:
        {
            std::vector<Elem> v2{5, 6, 9, 10};
            ViewPlainPtr referenceViewHost(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, referenceBufAcc, referenceViewHost, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(referenceBufAcc),
                    alpaka::test::mem::view::end(referenceBufAcc),
                    alpaka::test::mem::view::begin(subViewAcc)));
            alpaka::stream::enqueue(stream, compare);
            alpaka::wait::wait(stream);
            break;
        }
    case 3:
        {
            std::vector<Elem> v2{21, 22, 25, 26, 37, 38, 41, 42};
            ViewPlainPtr referenceViewHost(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, referenceBufAcc, referenceViewHost, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(referenceBufAcc),
                    alpaka::test::mem::view::end(referenceBufAcc),
                    alpaka::test::mem::view::begin(subViewAcc)));
            alpaka::stream::enqueue(stream, compare);
            alpaka::wait::wait(stream);
            break;
        }
    case 4:
        {
            /*
            std::vector<Elem> v2{75, 76, 78, 79, 91, 92, 95, 96, 139, 140, 143, 144, 155, 156, 159, 160};
            ViewPlainPtr referenceViewHost(v2.data(), devHost, extentView);
            alpaka::mem::view::copy(stream, referenceBufAcc, referenceViewHost, extentView);
            auto const compare(
                alpaka::exec::create<TAcc>(
                    workdiv,
                    compareBufferKernel,
                    alpaka::test::mem::view::begin(referenceBufAcc),
                    alpaka::test::mem::view::end(referenceBufAcc),
                    alpaka::test::mem::view::begin(subViewAcc)));
            */
            alpaka::wait::wait(stream);
            break;
        }
    default:
        alpaka::wait::wait(stream);
        break;
    };
}

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    copyViewSubViewGeneric,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;

    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;

    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;

    using View = alpaka::mem::view::ViewSubView<DevAcc, Elem, Dim, Size>;
    using ViewPlainPtr = alpaka::mem::view::ViewPlainPtr<DevHost, Elem, Dim, Size>;

#if BOOST_COMP_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4127) // conditional expression is constant
#endif

    if(Dim::value != 4)

#if BOOST_COMP_MSVC
    #pragma warning(pop)
#endif
    {
        DevHost const devHost (alpaka::pltf::getDevByIdx<PltfHost>(0));
        DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
        StreamAcc stream (devAcc);

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
