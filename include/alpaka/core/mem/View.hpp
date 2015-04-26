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

#include <alpaka/traits/Dim.hpp>        // DimT
#include <alpaka/traits/Dev.hpp>        // DevT
#include <alpaka/traits/Extent.hpp>     // traits::getXXX
#include <alpaka/traits/Offset.hpp>     // traits::getOffsetX
#include <alpaka/traits/mem/View.hpp>   // SpaceT, ...

#include <alpaka/core/Vec.hpp>          // Vec
#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST

namespace alpaka
{
    namespace mem
    {
        namespace detail
        {
            //#############################################################################
            //! A memory buffer view.
            //#############################################################################
            template<
                typename TBuf>
            class View
            {
            private:
                using Dim = dim::DimT<TBuf>;
                using Buf = BufT<dev::DevT<TBuf>, ElemT<TBuf>, Dim, SpaceT<TBuf>>;
                using MemSpace = SpaceT<TBuf>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer base or a memory buffer view itself.
                //-----------------------------------------------------------------------------
                View(
                    TBuf const & buf) :
                        m_Buf(getBuf(buf)),
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
                    typename TOffsets,
                    typename TExtents>
                View(
                    TBuf const & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(getBuf(buf)),
                        m_vOffsetsElements(offset::getOffsetsNd<Dim, UInt>(relativeOffsetsElements)+offset::getOffsetsNd<Dim, UInt>(buf)),
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
                typename TBuf>
            struct DevType<
                alpaka::mem::detail::View<TBuf>>
            {
                using type = alpaka::dev::DevT<TBuf>;
            };

            //#############################################################################
            //! The View device get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetDev<
                alpaka::mem::detail::View<TBuf>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    alpaka::mem::detail::View<TBuf> const & bufView)
                -> alpaka::dev::DevT<TBuf>
                {
                    return
                        alpaka::dev::getDev(
                            alpaka::mem::getBuf(bufView));
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The View dimension getter trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct DimType<
                alpaka::mem::detail::View<TBuf>>
            {
                using type = alpaka::dim::DimT<TBuf>;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The View width get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TBuf>
            struct GetExtent<
                TuiIdx,
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<alpaka::dim::DimT<TBuf>::value >= (TuiIdx+1)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    alpaka::mem::detail::View<TBuf> const & extents)
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
                typename TBuf>
            struct GetOffset<
                TuiIdx,
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<alpaka::dim::DimT<TBuf>::value >= (TuiIdx+1)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    alpaka::mem::detail::View<TBuf> const & offset)
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
                typename TBuf>
            struct SpaceType<
                alpaka::mem::detail::View<TBuf>>
            {
                using type = alpaka::mem::SpaceT<TBuf>;
            };

            //#############################################################################
            //! The View memory element type get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct ElemType<
                alpaka::mem::detail::View<TBuf>>
            {
                using type = alpaka::mem::ElemT<TBuf>;
            };

            //#############################################################################
            //! The View memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct ViewType<
                alpaka::mem::detail::View<TBuf>>
            {
                using type = alpaka::mem::ViewT<TBuf>;
            };

            //#############################################################################
            //! The memory buffer view creation type trait.
            //#############################################################################
            template<
                typename TBuf>
            struct CreateView<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto createView(
                    TBuf const & buf)
                -> alpaka::mem::detail::View<TBuf>
                {
                    return alpaka::mem::detail::View<TBuf>(
                        buf);
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents,
                    typename TOffsets>
                ALPAKA_FCT_HOST static auto createView(
                    TBuf const & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements)
                -> alpaka::mem::detail::View<TBuf>
                {
                    return alpaka::mem::detail::View<TBuf>(
                        buf,
                        extentsElements,
                        relativeOffsetsElements);
                }
            };

            //#############################################################################
            //! The View base buffer trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetBuf<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    alpaka::mem::detail::View<TBuf> const & bufView)
                -> TBuf const &
                {
                    return bufView.m_Buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    alpaka::mem::detail::View<TBuf> & bufView)
                -> TBuf &
                {
                    return bufView.m_Buf;
                }
            };

            //#############################################################################
            //! The View native pointer get trait specialization.
            // \TODO: Optimize by specializing per dim!
            //#############################################################################
            template<
                typename TBuf>
            struct GetNativePtr<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getNativePtr(
                    alpaka::mem::detail::View<TBuf> const & bufView)
                -> alpaka::mem::ElemT<TBuf> const *
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto const uiPitchElements(alpaka::mem::getPitchElements(bufView));
                    return alpaka::mem::getNativePtr(alpaka::mem::getBuf(bufView))
                        + alpaka::offset::getOffset<0u, UInt>(bufView)
                        + alpaka::offset::getOffset<1u, UInt>(bufView) * uiPitchElements
                        + alpaka::offset::getOffset<2u, UInt>(bufView) * uiPitchElements * alpaka::extent::getExtent<0u, UInt>(alpaka::mem::getBuf(bufView));
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getNativePtr(
                    alpaka::mem::detail::View<TBuf> & bufView)
                -> alpaka::mem::ElemT<TBuf> *
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto const uiPitchElements(alpaka::mem::getPitchElements(bufView));
                    return alpaka::mem::getNativePtr(alpaka::mem::getBuf(bufView))
                        + alpaka::offset::getOffset<0u, UInt>(bufView)
                        + alpaka::offset::getOffset<1u, UInt>(bufView) * uiPitchElements
                        + alpaka::offset::getOffset<2u, UInt>(bufView) * uiPitchElements * alpaka::extent::getExtent<0u, UInt>(alpaka::mem::getBuf(bufView));
                }
            };

            //#############################################################################
            //! The View pitch get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetPitchBytes<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPitchBytes(
                    alpaka::mem::detail::View<TBuf> const & bufView)
                -> UInt
                {
                    return alpaka::mem::getPitchElements(alpaka::mem::getBuf(bufView));
                }
            };
        }
    }
}
