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
                            extent::getExtentsVecNd<Dim, UInt>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsVecNd<Dim, UInt>(buf))
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
                            extent::getExtentsVecNd<Dim, UInt>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecNd<Dim, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsVecNd<Dim, UInt>(buf))
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
                            extent::getExtentsVecNd<Dim, UInt>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsVecNd<Dim, UInt>(extentsElements)),
                        m_vOffsetsElements(offset::getOffsetsVecNd<Dim, UInt>(relativeOffsetsElements) + offset::getOffsetsVecNd<Dim, UInt>(buf))
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
                            extent::getExtentsVecNd<Dim, UInt>(buf),
                            mem::view::getPitchBytes<Dim::value - 1u, UInt>(buf)),
                        m_vExtentsElements(extent::getExtentsVecNd<Dim, UInt>(extentsElements)),
                        m_vOffsetsElements(offset::getOffsetsVecNd<Dim, UInt>(relativeOffsetsElements) + offset::getOffsetsVecNd<Dim, UInt>(buf))
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
                -> UInt
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
                    using IdxSequence = detail::make_integer_sequence<UInt, TDim::value>;
                public:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrNative(
                        mem::view::ViewBasic<TElem, TDim, TDev> const & view)
                    -> TElem const *
                    {
                        auto const & buf(mem::view::getBuf(view));
                        // \TODO: Precalculate this pointer for faster execution.
                        return mem::view::getPtrNative(buf) + pitchedOffsetElems(view, buf, IdxSequence());
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrNative(
                        mem::view::ViewBasic<TElem, TDim, TDev> & view)
                    -> TElem *
                    {
                        auto & buf(mem::view::getBuf(view));
                        // \TODO: Precalculate this pointer for faster execution.
                        return mem::view::getPtrNative(buf) + pitchedOffsetElems(view, buf, IdxSequence());
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TView,
                        typename TBuf,
                        UInt... TIndices>
                    ALPAKA_FCT_HOST static auto pitchedOffsetElems(
                        TView const & view,
                        TBuf const & buf,
                        detail::integer_sequence<UInt, TIndices...> const &)
                    -> UInt
                    {
                        return
                            foldr(
                                std::plus<UInt>(),
                                pitchedOffsetElemsPerDim<TIndices>(view, buf)...);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        UInt TuiIdx,
                        typename TView,
                        typename TBuf>
                    ALPAKA_FCT_HOST static auto pitchedOffsetElemsPerDim(
                        TView const & view,
                        TBuf const & buf)
                    -> UInt
                    {
                        return
                            offset::getOffset<TuiIdx, UInt>(view)
                            * view::getPitchElements<TuiIdx + 1u, UInt>(buf);
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
                    -> UInt
                    {
                        return
                            view::getPitchElements<TIdx::value, UInt>(
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
                -> UInt
                {
                    return offset.m_vOffsetsElements[TIdx::value];
                }
            };
        }
    }
}
