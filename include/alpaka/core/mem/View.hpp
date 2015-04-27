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
#include <alpaka/traits/mem/View.hpp>               // SpaceT, ...

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
                using MemSpace = SpaceT<acc::AccT<TDev>>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer base or a memory buffer view itself.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                View(
                    TBuf const & buf) :
                        m_Buf(
                            mem::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsNd<Dim, UInt>(buf),
                            mem::getPitchBytes<0u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(buf))
                {}
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer base or a memory buffer view itself.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                View(
                    TBuf & buf) :
                        m_Buf(
                            mem::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsNd<Dim, UInt>(buf),
                            mem::getPitchBytes<0u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(buf))
                {}

                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer base or a memory buffer view itself.
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
                            mem::getPitchBytes<0u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(relativeOffsetsElements) + offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(extentsElements))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The base buffer and the extents are required to have the same dimensionality!");

                    assert(extent::getWidth<UInt>(relativeOffsetsElements) <= extent::getWidth<UInt>(buf));
                    assert(extent::getHeight<UInt>(relativeOffsetsElements) <= extent::getHeight<UInt>(buf));
                    assert(extent::getDepth<UInt>(relativeOffsetsElements) <= extent::getDepth<UInt>(buf));
                    assert((offset::getOffsetX<UInt>(relativeOffsetsElements)+offset::getOffsetX<UInt>(buf)+extent::getWidth<UInt>(extentsElements)) <= extent::getWidth<UInt>(buf));
                    assert((offset::getOffsetY<UInt>(relativeOffsetsElements)+offset::getOffsetY<UInt>(buf)+extent::getHeight<UInt>(extentsElements)) <= extent::getHeight<UInt>(buf));
                    assert((offset::getOffsetZ<UInt>(relativeOffsetsElements)+offset::getOffsetZ<UInt>(buf)+extent::getDepth<UInt>(extentsElements)) <= extent::getDepth<UInt>(buf));
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer base or a memory buffer view itself.
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
                            mem::getPitchBytes<0u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(relativeOffsetsElements) + offset::getOffsetsNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsNd<Dim, UInt>(extentsElements))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The base buffer and the extents are required to have the same dimensionality!");

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
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & bufView)
                -> TDev
                {
                    return
                        alpaka::dev::getDev(
                            alpaka::mem::getBase(bufView));
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
                UInt TuiIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetExtent<
                TuiIdx,
                alpaka::mem::detail::View<TElem, TDim, TDev>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & extents)
                -> UInt
                {
                    return extents.m_vExtentsElements[TuiIdx];
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The View x offset get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetOffset<
                TuiIdx,
                alpaka::mem::detail::View<TElem, TDim, TDev>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & offset)
                -> UInt
                {
                    return offset.m_vOffsetsElements[TuiIdx];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The View memory space trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct SpaceType<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                using type = typename alpaka::mem::detail::View<TElem, TDim, TDev>::MemSpace;
            };

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
            //! The View base trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetBase<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBase(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & bufView)
                -> typename alpaka::mem::detail::View<TElem, TDim, TDev>::Buf const &
                {
                    return bufView.m_Buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBase(
                    alpaka::mem::detail::View<TElem, TDim, TDev> & bufView)
                -> typename alpaka::mem::detail::View<TElem, TDim, TDev>::Buf &
                {
                    return bufView.m_Buf;
                }
            };

            //#############################################################################
            //! The View native pointer get trait specialization.
            // \TODO: Optimize by specializing per dim!
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetPtrNative<
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & bufView)
                -> TElem const *
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto const & buf(alpaka::mem::getBase(bufView));
                    auto const uiPitchElementsX(alpaka::mem::getPitchElements<0u, UInt>(buf));
                    return alpaka::mem::getPtrNative(buf)
                        + alpaka::offset::getOffset<0u, UInt>(bufView)
                        + alpaka::offset::getOffset<1u, UInt>(bufView) * uiPitchElementsX
                        + alpaka::offset::getOffset<2u, UInt>(bufView) * uiPitchElementsX * alpaka::mem::getPitchElements<1u, UInt>(buf);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    alpaka::mem::detail::View<TElem, TDim, TDev> & bufView)
                -> TElem *
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto & buf(alpaka::mem::getBase(bufView));
                    auto const uiPitchElementsX(alpaka::mem::getPitchElements<0u, UInt>(buf));
                    return alpaka::mem::getPtrNative(buf)
                        + alpaka::offset::getOffset<0u, UInt>(bufView)
                        + alpaka::offset::getOffset<1u, UInt>(bufView) * uiPitchElementsX
                        + alpaka::offset::getOffset<2u, UInt>(bufView) * uiPitchElementsX * alpaka::mem::getPitchElements<1u, UInt>(buf);
                }
            };

            //#############################################################################
            //! The View pitch get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetPitchBytes<
                TuiIdx,
                alpaka::mem::detail::View<TElem, TDim, TDev>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPitchBytes(
                    alpaka::mem::detail::View<TElem, TDim, TDev> const & bufView)
                -> UInt
                {
                    return
                        alpaka::mem::getPitchElements<TuiIdx, UInt>(
                            alpaka::mem::getBase(bufView));
                }
            };
        }
    }
}
