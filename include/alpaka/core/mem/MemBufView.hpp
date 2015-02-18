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

#include <alpaka/core/Common.hpp>           // ALPAKA_FCT_HOST
#include <alpaka/core/Vec.hpp>              // Vec

#include <alpaka/traits/Dim.hpp>            // DimT
#include <alpaka/traits/Extents.hpp>        // traits::getXXX
#include <alpaka/traits/Offsets.hpp>        // traits::getOffsetX
#include <alpaka/traits/mem/MemBufView.hpp> // MemSpaceT, ...

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
                typename TMemBuf>
            class MemBufView
            {
            private:
                using Dim = dim::DimT<TMemBuf>;
                using MemBufBase = alpaka::mem::MemBufBaseT<MemElemT<TMemBuf>, Dim, MemSpaceT<TMemBuf>>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param memBuf This can be either a memory buffer base or a memory buffer view itself.
                //-----------------------------------------------------------------------------
                MemBufView(
                    TMemBuf const & memBuf) :
                        m_memBufBase(alpaka::mem::getMemBufBase(memBuf)),
                        m_vOffsetsElements(Vec<Dim::value>::fromOffsets(memBuf)),
                        m_vExtentsElements(Vec<Dim::value>::fromExtents(memBuf))
                {}

                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param memBuf This can be either a memory buffer base or a memory buffer view itself.
                //! \param offsetsElements The offsets in elements.
                //! \param extentsElements The extents in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TOffsets,
                    typename TExtents>
                MemBufView(
                    TMemBuf const & memBuf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_memBufBase(alpaka::mem::getMemBufBase(memBuf)),
                        m_vOffsetsElements(Vec<Dim::value>::fromOffsets(relativeOffsetsElements)+Vec<Dim::value>::fromOffsets(memBuf)),
                        m_vExtentsElements(Vec<Dim::value>::fromExtents(extentsElements))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The base buffer and the extents are required to have the same dimensionality!");
                
                    assert(alpaka::extent::getWidth(relativeOffsetsElements) <= alpaka::extent::getWidth(memBuf));
                    assert(alpaka::extent::getHeight(relativeOffsetsElements) <= alpaka::extent::getHeight(memBuf));
                    assert(alpaka::extent::getDepth(relativeOffsetsElements) <= alpaka::extent::getDepth(memBuf));
                    assert((alpaka::offset::getOffsetX(relativeOffsetsElements)+alpaka::offset::getOffsetX(memBuf)+alpaka::extent::getWidth(extentsElements)) <= alpaka::extent::getWidth(memBuf));
                    assert((alpaka::offset::getOffsetY(relativeOffsetsElements)+alpaka::offset::getOffsetY(memBuf)+alpaka::extent::getHeight(extentsElements)) <= alpaka::extent::getHeight(memBuf));
                    assert((alpaka::offset::getOffsetZ(relativeOffsetsElements)+alpaka::offset::getOffsetZ(memBuf)+alpaka::extent::getDepth(extentsElements)) <= alpaka::extent::getDepth(memBuf));
                }

            private:
                MemBufBase m_memBufBase;
                Vec<Dim::value> m_vOffsetsElements;
                Vec<Dim::value> m_vExtentsElements;
            };
        }
    }

    
    //-----------------------------------------------------------------------------
    // Trait specializations for MemBufView.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufView dimension getter trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct DimType<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                using type = alpaka::dim::DimT<TMemBuf>;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufView width get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetWidth<
                alpaka::mem::detail::MemBufView<TMemBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TMemBuf>::value >= 1u) && (alpaka::dim::DimT<TMemBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getWidth(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & extent)
                {
                    return extent.m_vExtentsElements[0u];
                }
            };

            //#############################################################################
            //! The MemBufView height get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetHeight<
                alpaka::mem::detail::MemBufView<TMemBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TMemBuf>::value >= 2u) && (alpaka::dim::DimT<TMemBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getHeight(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & extent)
                {
                    return extent.m_vExtentsElements[1u];
                }
            };
            //#############################################################################
            //! The MemBufView depth get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetDepth<
                alpaka::mem::detail::MemBufView<TMemBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TMemBuf>::value >= 3u) && (alpaka::dim::DimT<TMemBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDepth(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & extent)
                {
                    return extent.m_vExtentsElements[2u];
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The MemBufView offsets get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetOffsets<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static Vec<alpaka::dim::DimT<TMemBuf>::value> getOffsets(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & offsets)
                {
                    return offsets.m_vOffsetsElements;
                }
            };

            //#############################################################################
            //! The MemBufView x offset get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetOffsetX<
                alpaka::mem::detail::MemBufView<TMemBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TMemBuf>::value >= 1u) && (alpaka::dim::DimT<TMemBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getOffsetX(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & offset)
                {
                    return offset.m_vOffsetsElements[0u];
                }
            };

            //#############################################################################
            //! The MemBufView y offset get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetOffsetY<
                alpaka::mem::detail::MemBufView<TMemBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TMemBuf>::value >= 2u) && (alpaka::dim::DimT<TMemBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getOffsetY(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & offset)
                {
                    return offset.m_vOffsetsElements[1u];
                }
            };
            //#############################################################################
            //! The MemBufView z offset get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetOffsetZ<
                alpaka::mem::detail::MemBufView<TMemBuf>,
                typename std::enable_if<(alpaka::dim::DimT<TMemBuf>::value >= 3u) && (alpaka::dim::DimT<TMemBuf>::value <= 3u)>::type>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getOffsetZ(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & offset)
                {
                    return offset.m_vOffsetsElements[2u];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The MemBufView memory space trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct MemSpaceType<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                using type = alpaka::mem::MemSpaceT<TMemBuf>;
            };

            //#############################################################################
            //! The MemBufView memory element type get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct MemElemType<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                using type = alpaka::mem::MemElemT<TMemBuf>;
            };

            //#############################################################################
            //! The MemBufBaseCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct MemBufViewType<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                using type = alpaka::mem::MemBufViewT<TMemBuf>;
            };

            //#############################################################################
            //! The MemBufView base buffer trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetMemBufBase<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TMemBuf const & getMemBufBase(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & memBufView)
                {
                    return memBufView.m_memBufBase;
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static TMemBuf & getMemBufBase(
                    alpaka::mem::detail::MemBufView<TMemBuf> & memBufView)
                {
                    return memBufView.m_memBufBase;
                }
            };

            //#############################################################################
            //! The MemBufView native pointer get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetNativePtr<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::MemElemT<TMemBuf> const * getNativePtr(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & memBufView)
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto const uiPitchElements(alpaka::mem::getPitchElements(memBufView));
                    return alpaka::mem::getNativePtr(alpaka::mem::getMemBufBase(memBufView))
                        + alpaka::offset::getOffsetX(memBufView)
                        + alpaka::offset::getOffsetY(memBufView) * uiPitchElements
                        + alpaka::offset::getOffsetZ(memBufView) * uiPitchElements * alpaka::extent::getHeight(alpaka::mem::getMemBufBase(memBufView));
                }
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static alpaka::mem::MemElemT<TMemBuf> * getNativePtr(
                    alpaka::mem::detail::MemBufView<TMemBuf> & memBufView)
                {
                    // \TODO: Precalculate this pointer for faster execution.
                    auto const uiPitchElements(alpaka::mem::getPitchElements(memBufView));
                    return alpaka::mem::getNativePtr(alpaka::mem::getMemBufBase(memBufView))
                        + alpaka::offset::getOffsetX(memBufView)
                        + alpaka::offset::getOffsetY(memBufView) * uiPitchElements
                        + alpaka::offset::getOffsetZ(memBufView) * uiPitchElements * alpaka::extent::getHeight(alpaka::mem::getMemBufBase(memBufView));
                }
            };

            //#############################################################################
            //! The CUDA buffer pitch get trait specialization.
            //#############################################################################
            template<
                typename TMemBuf>
            struct GetPitchBytes<
                alpaka::mem::detail::MemBufView<TMemBuf>>
            {
                //-----------------------------------------------------------------------------
                //! 
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getPitchBytes(
                    alpaka::mem::detail::MemBufView<TMemBuf> const & memBufView)
                {
                    return alpaka::mem::getPitchElements(alpaka::mem::getMemBufBase(memBufView));
                }
            };
        }
    }
}
