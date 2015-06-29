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

#include <alpaka/dim/Traits.hpp>                    // DimT
#include <alpaka/dev/Traits.hpp>                    // DevT
#include <alpaka/extent/Traits.hpp>                 // view::getXXX
#include <alpaka/offset/Traits.hpp>                 // traits::getOffsetX
#include <alpaka/mem/view/Traits.hpp>

#include <alpaka/mem/buf/BufPlainPtrWrapper.hpp>    // BufPlainPtrWrapper
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
        namespace view
        {
            //#############################################################################
            //! A memory buffer view.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            class ViewBasic
            {
            public:
                using Elem = TElem;
                using Dim = TDim;
                using Dev = TDev;
                using Buf = buf::BufPlainPtrWrapper<TElem, TDim, TDev>;
                // If the value type is const, we store a const buffer.
                //using BufC = detail::MimicConst<TElem, Buf>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                ViewBasic(
                    TBuf const & buf) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsVecEnd<Dim, Uint>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, Uint>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<Dim, Uint>(buf)),
                        m_vExtentsElements(extent::getExtentsVecEnd<Dim, Uint>(buf))
                {}
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                ViewBasic(
                    TBuf & buf) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsVecEnd<Dim, Uint>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, Uint>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<Dim, Uint>(buf)),
                        m_vExtentsElements(extent::getExtentsVecEnd<Dim, Uint>(buf))
                {}

                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //! \param extentsElements The extents in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TOffsets,
                    typename TExtents>
                ViewBasic(
                    TBuf const & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsVecEnd<Dim, Uint>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, Uint>(buf)),
                        m_vExtentsElements(extent::getExtentsVecEnd<Dim, Uint>(extentsElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<Dim, Uint>(relativeOffsetsElements) + offset::getOffsetsVecEnd<Dim, Uint>(buf))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The buffer and the extents are required to have the same dimensionality!");

                    assert(extent::getWidth<Uint>(relativeOffsetsElements) <= extent::getWidth<Uint>(buf));
                    assert(extent::getHeight<Uint>(relativeOffsetsElements) <= extent::getHeight<Uint>(buf));
                    assert(extent::getDepth<Uint>(relativeOffsetsElements) <= extent::getDepth<Uint>(buf));
                    assert((offset::getOffsetX<Uint>(relativeOffsetsElements)+offset::getOffsetX<Uint>(buf)+extent::getWidth<Uint>(extentsElements)) <= extent::getWidth<Uint>(buf));
                    assert((offset::getOffsetY<Uint>(relativeOffsetsElements)+offset::getOffsetY<Uint>(buf)+extent::getHeight<Uint>(extentsElements)) <= extent::getHeight<Uint>(buf));
                    assert((offset::getOffsetZ<Uint>(relativeOffsetsElements)+offset::getOffsetZ<Uint>(buf)+extent::getDepth<Uint>(extentsElements)) <= extent::getDepth<Uint>(buf));
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //! \param extentsElements The extents in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TOffsets,
                    typename TExtents>
                ViewBasic(
                    TBuf & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsVecEnd<Dim, Uint>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, Uint>(buf)),
                        m_vExtentsElements(extent::getExtentsVecEnd<Dim, Uint>(extentsElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<Dim, Uint>(relativeOffsetsElements) + offset::getOffsetsVecEnd<Dim, Uint>(buf))
                {
                    static_assert(
                        std::is_same<Dim, dim::DimT<TExtents>>::value,
                        "The buffer and the extents are required to have the same dimensionality!");

                    assert(extent::getWidth<Uint>(relativeOffsetsElements) <= extent::getWidth<Uint>(buf));
                    assert(extent::getHeight<Uint>(relativeOffsetsElements) <= extent::getHeight<Uint>(buf));
                    assert(extent::getDepth<Uint>(relativeOffsetsElements) <= extent::getDepth<Uint>(buf));
                    assert((offset::getOffsetX<Uint>(relativeOffsetsElements)+offset::getOffsetX<Uint>(buf)+extent::getWidth<Uint>(extentsElements)) <= extent::getWidth<Uint>(buf));
                    assert((offset::getOffsetY<Uint>(relativeOffsetsElements)+offset::getOffsetY<Uint>(buf)+extent::getHeight<Uint>(extentsElements)) <= extent::getHeight<Uint>(buf));
                    assert((offset::getOffsetZ<Uint>(relativeOffsetsElements)+offset::getOffsetZ<Uint>(buf)+extent::getDepth<Uint>(extentsElements)) <= extent::getDepth<Uint>(buf));
                }

            public:
                Buf m_Buf;
                Vec<Dim> m_vExtentsElements;
                Vec<Dim> m_vOffsetsElements;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for ViewBasic.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewBasic device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DevType<
                mem::view::ViewBasic<TElem, TDim, TDev>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The ViewBasic device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetDev<
                mem::view::ViewBasic<TElem, TDim, TDev>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    mem::view::ViewBasic<TElem, TDim, TDev> const & view)
                -> TDev
                {
                    return
                        dev::getDev(
                            mem::view::getBuf(view));
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewBasic dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev>
            struct DimType<
                mem::view::ViewBasic<TElem, TDim, TDev>>
            {
                using type = TDim;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewBasic width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetExtent<
                TIdx,
                mem::view::ViewBasic<TElem, TDim, TDev>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    mem::view::ViewBasic<TElem, TDim, TDev> const & extents)
                -> Uint
                {
                    return extents.m_vExtentsElements[TIdx::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The ViewBasic memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct ElemType<
                    mem::view::ViewBasic<TElem, TDim, TDev>>
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
                    mem::view::ViewBasic<TElem, TDim, TDev>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBuf>
                    ALPAKA_FCT_HOST static auto createView(
                        TBuf const & buf)
                    -> mem::view::ViewBasic<typename std::add_const<TElem>::type, TDim, TDev>
                    {
                        return mem::view::ViewBasic<
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
                    -> mem::view::ViewBasic<TElem, TDim, TDev>
                    {
                        return mem::view::ViewBasic<
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
                    -> mem::view::ViewBasic<typename std::add_const<TElem>::type, TDim, TDev>
                    {
                        return mem::view::ViewBasic<
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
                    -> mem::view::ViewBasic<TElem, TDim, TDev>
                    {
                        return mem::view::ViewBasic<
                            TElem,
                            TDim,
                            TDev>(
                                buf,
                                extentsElements,
                                relativeOffsetsElements);
                    }
                };

                //#############################################################################
                //! The ViewBasic buf trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct GetBuf<
                    mem::view::ViewBasic<TElem, TDim, TDev>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getBuf(
                        mem::view::ViewBasic<TElem, TDim, TDev> const & view)
                    -> typename mem::view::ViewBasic<TElem, TDim, TDev>::Buf const &
                    {
                        return view.m_Buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getBuf(
                        mem::view::ViewBasic<TElem, TDim, TDev> & view)
                    -> typename mem::view::ViewBasic<TElem, TDim, TDev>::Buf &
                    {
                        return view.m_Buf;
                    }
                };

                //#############################################################################
                //! The ViewBasic native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct GetPtrNative<
                    mem::view::ViewBasic<TElem, TDim, TDev>>
                {
                private:
                    using IdxSequence = alpaka::detail::make_integer_sequence<Uint, TDim::value>;
                public:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrNative(
                        mem::view::ViewBasic<TElem, TDim, TDev> const & view)
                    -> TElem const *
                    {
                        auto const & buf(mem::view::getBuf(view));
                        // \TODO: pre-calculate this pointer for faster execution.
                        return
                            reinterpret_cast<TElem const *>(
                                reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(buf))
                                + pitchedOffsetBytes(view, buf, IdxSequence()));
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrNative(
                        mem::view::ViewBasic<TElem, TDim, TDev> & view)
                    -> TElem *
                    {
                        auto & buf(mem::view::getBuf(view));
                        // \TODO: pre-calculate this pointer for faster execution.
                        return
                            reinterpret_cast<TElem *>(
                                reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(buf))
                                + pitchedOffsetBytes(view, buf, IdxSequence()));
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TView,
                        typename TBuf,
                        Uint... TIndices>
                    ALPAKA_FCT_HOST static auto pitchedOffsetBytes(
                        TView const & view,
                        TBuf const & buf,
                        alpaka::detail::integer_sequence<Uint, TIndices...> const &)
                    -> Uint
                    {
                        return
                            foldr(
                                std::plus<Uint>(),
                                pitchedOffsetBytesDim<TIndices>(view, buf)...);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        Uint TuiIdx,
                        typename TView,
                        typename TBuf>
                    ALPAKA_FCT_HOST static auto pitchedOffsetBytesDim(
                        TView const & view,
                        TBuf const & buf)
                    -> Uint
                    {
                        return
                            offset::getOffset<TuiIdx, Uint>(view)
                            * view::getPitchBytes<TuiIdx + 1u, Uint>(buf);
                    }
                };

                //#############################################################################
                //! The ViewBasic pitch get trait specialization.
                //#############################################################################
                template<
                    typename TIdx,
                    typename TElem,
                    typename TDim,
                    typename TDev>
                struct GetPitchBytes<
                    TIdx,
                    mem::view::ViewBasic<TElem, TDim, TDev>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPitchBytes(
                        mem::view::ViewBasic<TElem, TDim, TDev> const & view)
                    -> Uint
                    {
                        return
                            view::getPitchBytes<TIdx::value, Uint>(
                                mem::view::getBuf(view));
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewBasic x offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev>
            struct GetOffset<
                TIdx,
                mem::view::ViewBasic<TElem, TDim, TDev>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    mem::view::ViewBasic<TElem, TDim, TDev> const & offset)
                -> Uint
                {
                    return offset.m_vOffsetsElements[TIdx::value];
                }
            };
        }
    }
}
