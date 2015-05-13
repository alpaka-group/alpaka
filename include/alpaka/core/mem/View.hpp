/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#pragma once

#include <alpaka/traits/Dim.hpp>                    // DimT
#include <alpaka/traits/Dev.hpp>                    // DevT
#include <alpaka/traits/Extent.hpp>                 // traits::getXXX
#include <alpaka/traits/Offset.hpp>                 // traits::getOffsetX
#include <alpaka/traits/mem/View.hpp>

#include <alpaka/core/mem/BufPlainPtrWrapper.hpp>   // BufPlainPtrWrapper
#include <alpaka/core/Vec.hpp>                      // Vec
#include <alpaka/core/Common.hpp>                   // ALPAKA_FCT_HOST

#include <type_traits>                              // std::conditional, ...

namespace alpaka
{
    /*namespace detail
    {
        //#############################################################################
        // \tparam TSource Type to mimic the constness of.
        // \tparam T Type to conditionally make const.
        //#############################################################################
        template<
            typename TSource,
            typename T>
        using MimicConst = typename std::conditional<
            std::is_const<TSource>::value,
            typename std::add_const<T>::type,
            typename std::remove_const<T>::type>;
    }*/
    namespace mem
    {
        namespace detail
        {
            //#############################################################################
            //! A memory buffer view.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            class View
            {
            public:
                using Elem = TElem;
                using Dim = TDim;
                using Dev = TDev;
                using Buf = BufPlainPtrWrapper<TElem, TDim, TDev>;
                // If the value type is const, we store a const buffer.
                //using BufC = alpaka::detail::MimicConst<TElem, Buf>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                View(
                    TBuf const & buf) :
                        m_Buf(
                            mem::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsNd<Dim, UInt>(buf),
                            mem::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(buf))
                {}
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                View(
                    TBuf & buf) :
                        m_Buf(
                            mem::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsNd<Dim, UInt>(buf),
                            mem::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(buf))
                {}

                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //! \param offsetsElements The offsets in elements.
                //! \param extentsElements The extents in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TOffsets,
                    typename TExtents>
                View(
                    TBuf const & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsNd<Dim, UInt>(buf),
                            mem::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(relativeOffsetsElements) + offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(extentsElements))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The buffer and the extents are required to have the same dimensionality!");

                    assert(extent::getWidth<UInt>(relativeOffsetsElements) <= extent::getWidth<UInt>(buf));
                    assert(extent::getHeight<UInt>(relativeOffsetsElements) <= extent::getHeight<UInt>(buf));
                    assert(extent::getDepth<UInt>(relativeOffsetsElements) <= extent::getDepth<UInt>(buf));
                    assert((offset::getOffsetX<UInt>(relativeOffsetsElements)+offset::getOffsetX<UInt>(buf)+extent::getWidth<UInt>(extentsElements)) <= extent::getWidth<UInt>(buf));
                    assert((offset::getOffsetY<UInt>(relativeOffsetsElements)+offset::getOffsetY<UInt>(buf)+extent::getHeight<UInt>(extentsElements)) <= extent::getHeight<UInt>(buf));
                    assert((offset::getOffsetZ<UInt>(relativeOffsetsElements)+offset::getOffsetZ<UInt>(buf)+extent::getDepth<UInt>(extentsElements)) <= extent::getDepth<UInt>(buf));
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //! \param offsetsElements The offsets in elements.
                //! \param extentsElements The extents in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TOffsets,
                    typename TExtents>
                View(
                    TBuf & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsNd<Dim, UInt>(buf),
                            mem::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(relativeOffsetsElements) + offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(extentsElements))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The buffer and the extents are required to have the same dimensionality!");

                    assert(extent::getWidth<UInt>(relativeOffsetsElements) <= extent::getWidth<UInt>(buf));
                    assert(extent::getHeight<UInt>(relativeOffsetsElements) <= extent::getHeight<UInt>(buf));
                    assert(extent::getDepth<UInt>(relativeOffsetsElements) <= extent::getDepth<UInt>(buf));
                    assert((offset::getOffsetX<UInt>(relativeOffsetsElements)+offset::getOffsetX<UInt>(buf)+extent::getWidth<UInt>(extentsElements)) <= extent::getWidth<UInt>(buf));
                    assert((offset::getOffsetY<UInt>(relativeOffsetsElements)+offset::getOffsetY<UInt>(buf)+extent::getHeight<UInt>(extentsElements)) <= extent::getHeight<UInt>(buf));
                    assert((offset::getOffsetZ<UInt>(relativeOffsetsElements)+offset::getOffsetZ<UInt>(buf)+extent::getDepth<UInt>(extentsElements)) <= extent::getDepth<UInt>(buf));
                }

            public:
                Buf m_Buf;
                Vec<Dim> m_vOffsetsElements;
                Vec<Dim> m_vExtentsElements;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for View.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The View device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DevType<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The View device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetDev<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & view)
                -> TDev
                {
                    return
                        alpaka::dev::getDev(
                            alpaka::mem::getBuf(view));
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The View dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DimType<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The View width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetExtent<
                TIdx,
                alpaka::mem::detail::View<TElem, TDim, TDev>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & extents)
                -> UInt
                {
                    return extents.m_vExtentsElements[TIdx::value];
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The View x offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetOffset<
                TIdx,
                alpaka::mem::detail::View<TElem, TDim, TDev>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & offset)
                -> UInt
                {
                    return offset.m_vOffsetsElements[TIdx::value];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The View memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct ElemType<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The memory buffer view creation type trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct CreateView<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                ALPAKA_FCT_HOST static auto createView(
                    TBuf const & buf)
                -> alpaka::mem::detail::View<typename std::add_const<TElem>::type, TDim, TDev>
                {
                    return alpaka::mem::detail::View<
                        typename std::add_const<TElem>::type,
                        TDim,
                        TDev>(
                            buf);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                ALPAKA_FCT_HOST static auto createView(
                    TBuf & buf)
                -> alpaka::mem::detail::View<TElem, TDim, TDev>
                {
                    return alpaka::mem::detail::View<
                        TElem,
                        TDim,
                        TDev>(
                            buf);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TExtents,
                    typename TOffsets>
                ALPAKA_FCT_HOST static auto createView(
                    TBuf const & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements)
                -> alpaka::mem::detail::View<typename std::add_const<TElem>::type, TDim, TDev>
                {
                    return alpaka::mem::detail::View<
                        typename std::add_const<TElem>::type,
                        TDim,
                        TDev>(
                            buf,
                            extentsElements,
                            relativeOffsetsElements);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TExtents,
                    typename TOffsets>
                ALPAKA_FCT_HOST static auto createView(
                    TBuf & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements)
                -> alpaka::mem::detail::View<TElem, TDim, TDev>
                {
                    return alpaka::mem::detail::View<
                        TElem,
                        TDim,
                        TDev>(
                            buf,
                            extentsElements,
                            relativeOffsetsElements);
                }
            };

            //#############################################################################
            //! The View buf trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetBuf<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & view)
                -> typename alpaka::mem::detail::View<TElem, TDim, TDev>::Buf const &
                {
                    return view.m_Buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    alpaka::mem::detail::View<TElem, TDim, TDev> & view)
                -> typename alpaka::mem::detail::View<TElem, TDim, TDev>::Buf &
                {
                    return view.m_Buf;
                }
            };

            //#############################################################################
            //! The View native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetPtrNative<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
            private:
#if (BOOST_COMP_MSVC) && (BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14, 0, 0))
                using IdxSequence = typename alpaka::detail::make_integer_sequence<UInt, TDim::value>::type;
#else
                using IdxSequence = alpaka::detail::make_integer_sequence<UInt, TDim::value>;
#endif
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & view)
                -> TElem const *
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    return getPtrNativeInternal(
                        view,
                        IdxSequence());
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    alpaka::mem::detail::View<TElem, TDim, TDev> & view)
                -> TElem *
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    return getPtrNativeInternal(
                        view,
                        IdxSequence());
                }

            private:
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TView,
                    UInt... TIndices>
                ALPAKA_FCT_HOST static auto getPtrNativeInternal(
                    TView && view,
                    alpaka::detail::integer_sequence<UInt, TIndices...> const &)
                -> TElem *
                {
                    auto & buf(alpaka::mem::getBuf(view));
                    return alpaka::mem::getPtrNative(buf)
                        + alpaka::foldr(
                            std::plus<UInt>(),
                            basePtrOffsetElems<TIndices>(view, buf)...);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TIdx,
                    typename TView,
                    typename TBuf>
                ALPAKA_FCT_HOST static auto basePtrOffsetElems(
                    TView const & view,
                    TBuf const & buf)
                -> UInt
                {
                    return
                        alpaka::offset::getOffset<TIdx::value, UInt>(view)
                        * alpaka::mem::getPitchElements<TIdx::value + 1u, UInt>(buf);
                }
            };

            //#############################################################################
            //! The View pitch get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetPitchBytes<
                TIdx,
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPitchBytes(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & view)
                -> UInt
                {
                    return
                        alpaka::mem::getPitchElements<TIdx, UInt>(
                            alpaka::mem::getBuf(view));
                }
            };
        }
    }
}
