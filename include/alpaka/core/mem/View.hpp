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

#include <alpaka/core/Common.hpp>       // ALPAKA_FCT_HOST
#include <alpaka/core/Vec.hpp>          // Vec

#include <alpaka/traits/Dim.hpp>        // DimT
#include <alpaka/traits/Extents.hpp>    // traits::getXXX
#include <alpaka/traits/Offsets.hpp>    // traits::getOffsetX
#include <alpaka/traits/mem/View.hpp>   // SpaceT, ...

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
                using Buf = BufT<ElemT<TBuf>, Dim, SpaceT<TBuf>>;
                using MemSpace = SpaceT<TBuf>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer base or a memory buffer view itself.
                //-----------------------------------------------------------------------------
                View(
                    TBuf const & buf) :
                        m_buf(getBuf(buf)),
                        m_vOffsetsElements(Vec<Dim::value>::fromOffsets(buf)),
                        m_vExtentsElements(Vec<Dim::value>::fromExtents(buf))
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
                        m_buf(getBuf(buf)),
                        m_vOffsetsElements(Vec<Dim::value>::fromOffsets(relativeOffsetsElements)+Vec<Dim::value>::fromOffsets(buf)),
                        m_vExtentsElements(Vec<Dim::value>::fromExtents(extentsElements))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The base buffer and the extents are required to have the same dimensionality!");
                
                    assert(extent::getWidth(relativeOffsetsElements) <= extent::getWidth(buf));
                    assert(extent::getHeight(relativeOffsetsElements) <= extent::getHeight(buf));
                    assert(extent::getDepth(relativeOffsetsElements) <= extent::getDepth(buf));
                    assert((offset::getOffsetX(relativeOffsetsElements)+offset::getOffsetX(buf)+extent::getWidth(extentsElements)) <= extent::getWidth(buf));
                    assert((offset::getOffsetY(relativeOffsetsElements)+offset::getOffsetY(buf)+extent::getHeight(extentsElements)) <= extent::getHeight(buf));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+offset::getOffsetZ(buf)+extent::getDepth(extentsElements)) <= extent::getDepth(buf));
                }
                
            public:
                Buf m_buf;
                Vec<Dim::value> m_vOffsetsElements;
                Vec<Dim::value> m_vExtentsElements;
            };
        }
    }

    
    //-----------------------------------------------------------------------------
    // Trait specializations for View.
    //-----------------------------------------------------------------------------
    namespace traits
    {
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
            //! The View extents get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetExtents<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<alpaka::dim::DimT<TBuf>::value> getExtents(
                    alpaka::mem::detail::View<TBuf> const & extents)
                {
                    return {extents.m_vExtentsElements};
                }
            };

            //#############################################################################
            //! The View width get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetWidth<
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TBuf>::value >= 1u) && (alpaka::dim::DimT<TBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getWidth(
                    alpaka::mem::detail::View<TBuf> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The View height get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetHeight<
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TBuf>::value >= 2u) && (alpaka::dim::DimT<TBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getHeight(
                    alpaka::mem::detail::View<TBuf> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The View depth get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetDepth<
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TBuf>::value >= 3u) && (alpaka::dim::DimT<TBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getDepth(
                    alpaka::mem::detail::View<TBuf> const & extent)
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The View offsets get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetOffsets<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<alpaka::dim::DimT<TBuf>::value> getOffsets(
                    alpaka::mem::detail::View<TBuf> const & offsets)
                {
                    return offsets.m_vOffsetsElements;
                }
            };

            //#############################################################################
            //! The View x offset get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetOffsetX<
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TBuf>::value >= 1u) && (alpaka::dim::DimT<TBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getOffsetX(
                    alpaka::mem::detail::View<TBuf> const & offset)
                {
                    return offset.m_vOffsetsElements[0u];
                }
            };

            //#############################################################################
            //! The View y offset get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetOffsetY<
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TBuf>::value >= 2u) && (alpaka::dim::DimT<TBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getOffsetY(
                    alpaka::mem::detail::View<TBuf> const & offset)
                {
                    return offset.m_vOffsetsElements[1u];
                }
            };
            //#############################################################################
            //! The View z offset get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetOffsetZ<
                alpaka::mem::detail::View<TBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TBuf>::value >= 3u) && (alpaka::dim::DimT<TBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getOffsetZ(
                    alpaka::mem::detail::View<TBuf> const & offset)
                {
                    return offset.m_vOffsetsElements[2u];
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
            //! The BufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct ViewType<
                alpaka::mem::detail::View<TBuf>>
            {
                using type = alpaka::mem::ViewT<TBuf>;
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
                ALPAKA_FCT_HOST static TBuf const & getBuf(
                    alpaka::mem::detail::View<TBuf> const & memBufView)
                {
                    return memBufView.m_buf;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TBuf & getBuf(
                    alpaka::mem::detail::View<TBuf> & memBufView)
                {
                    return memBufView.m_buf;
                }
            };

            //#############################################################################
            //! The View native pointer get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetNativePtr<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::ElemT<TBuf> const * getNativePtr(
                    alpaka::mem::detail::View<TBuf> const & memBufView)
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto const uiPitchElements(alpaka::mem::getPitchElements(memBufView));
                    return alpaka::mem::getNativePtr(alpaka::mem::getBuf(memBufView))
                        + alpaka::offset::getOffsetX(memBufView)
                        + alpaka::offset::getOffsetY(memBufView) * uiPitchElements
                        + alpaka::offset::getOffsetZ(memBufView) * uiPitchElements * alpaka::extent::getHeight(alpaka::mem::getBuf(memBufView));
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::ElemT<TBuf> * getNativePtr(
                    alpaka::mem::detail::View<TBuf> & memBufView)
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto const uiPitchElements(alpaka::mem::getPitchElements(memBufView));
                    return alpaka::mem::getNativePtr(alpaka::mem::getBuf(memBufView))
                        + alpaka::offset::getOffsetX(memBufView)
                        + alpaka::offset::getOffsetY(memBufView) * uiPitchElements
                        + alpaka::offset::getOffsetZ(memBufView) * uiPitchElements * alpaka::extent::getHeight(alpaka::mem::getBuf(memBufView));
                }
            };

            //#############################################################################
            //! The CUDA buffer pitch get trait specialization.
            //#############################################################################
            template<
                typename TBuf>
            struct GetPitchBytes<
                alpaka::mem::detail::View<TBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static UInt getPitchBytes(
                    alpaka::mem::detail::View<TBuf> const & memBufView)
                {
                    return alpaka::mem::getPitchElements(alpaka::mem::getBuf(memBufView));
                }
            };
        }
    }
}
