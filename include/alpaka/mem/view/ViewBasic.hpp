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

#include <alpaka/dim/Traits.hpp>                    // Dim
#include <alpaka/dev/Traits.hpp>                    // Dev
#include <alpaka/extent/Traits.hpp>                 // view::getXXX
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/offset/Traits.hpp>                 // traits::getOffsetX
#include <alpaka/size/Traits.hpp>                   // size::traits::SizeType

#include <alpaka/mem/buf/BufPlainPtrWrapper.hpp>    // BufPlainPtrWrapper
#include <alpaka/vec/Vec.hpp>                       // Vec
#include <alpaka/core/Common.hpp>                   // ALPAKA_FN_HOST

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
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            class ViewBasic
            {
            public:
                using Dev = TDev;
                using Elem = TElem;
                using Dim = TDim;
                using Buf = buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>;
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
                            extent::getExtentsVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(buf)),
                        m_extentsElements(extent::getExtentsVecEnd<TDim>(buf))
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
                            extent::getExtentsVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(buf)),
                        m_extentsElements(extent::getExtentsVecEnd<TDim>(buf))
                {
                    static_assert(
                        std::is_same<TSize, size::Size<TBuf>>::value,
                        "The size type of TBuf and the TSize template parameter have to be identical!");
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
                    TBuf const & buf,
                    TExtents const & extentsElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentsVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_extentsElements(extent::getExtentsVecEnd<TDim>(extentsElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(relativeOffsetsElements) + offset::getOffsetsVecEnd<TDim>(buf))
                {
                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtents>>::value,
                        "The buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtents>>::value,
                        "The size type of TExtents and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TBuf>>::value,
                        "The size type of TBuf and the TSize template parameter have to be identical!");

                    assert(extent::getWidth(relativeOffsetsElements) <= extent::getWidth(buf));
                    assert(extent::getHeight(relativeOffsetsElements) <= extent::getHeight(buf));
                    assert(extent::getDepth(relativeOffsetsElements) <= extent::getDepth(buf));
                    assert((offset::getOffsetX(relativeOffsetsElements)+offset::getOffsetX(buf)+extent::getWidth(extentsElements)) <= extent::getWidth(buf));
                    assert((offset::getOffsetY(relativeOffsetsElements)+offset::getOffsetY(buf)+extent::getHeight(extentsElements)) <= extent::getHeight(buf));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+offset::getOffsetZ(buf)+extent::getDepth(extentsElements)) <= extent::getDepth(buf));
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
                            extent::getExtentsVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_extentsElements(extent::getExtentsVecEnd<TDim>(extentsElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(relativeOffsetsElements) + offset::getOffsetsVecEnd<TDim>(buf))
                {
                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtents>>::value,
                        "The buffer and the extents are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtents>>::value,
                        "The size type of TExtents and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TBuf>>::value,
                        "The size type of TBuf and the TSize template parameter have to be identical!");

                    assert(extent::getWidth(relativeOffsetsElements) <= extent::getWidth(buf));
                    assert(extent::getHeight(relativeOffsetsElements) <= extent::getHeight(buf));
                    assert(extent::getDepth(relativeOffsetsElements) <= extent::getDepth(buf));
                    assert((offset::getOffsetX(relativeOffsetsElements)+offset::getOffsetX(buf)+extent::getWidth(extentsElements)) <= extent::getWidth(buf));
                    assert((offset::getOffsetY(relativeOffsetsElements)+offset::getOffsetY(buf)+extent::getHeight(extentsElements)) <= extent::getHeight(buf));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+offset::getOffsetZ(buf)+extent::getDepth(extentsElements)) <= extent::getDepth(buf));
                }

            public:
                Buf m_Buf;
                Vec<TDim, TSize> m_extentsElements;
                Vec<TDim, TSize> m_vOffsetsElements;
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
                typename TDev,
                typename TSize>
            struct DevType<
                mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The ViewBasic device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct GetDev<
                mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize> const & view)
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
                typename TDev,
                typename TSize>
            struct DimType<
                mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
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
                typename TDev,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::view::ViewBasic<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize> const & extents)
                -> TSize
                {
                    return extents.m_extentsElements[TIdx::value];
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
                    typename TDev,
                    typename TSize>
                struct ElemType<
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
                {
                    using type = TElem;
                };

                //#############################################################################
                //! The memory buffer view creation type trait.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev,
                    typename TSize>
                struct CreateView<
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBuf>
                    ALPAKA_FN_HOST static auto createView(
                        TBuf const & buf)
                    -> mem::view::ViewBasic<typename std::add_const<TElem>::type, TDim, TDev, TSize>
                    {
                        return mem::view::ViewBasic<
                            TDev,
                            typename std::add_const<TElem>::type,
                            TDim,
                            TSize>(
                                buf);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBuf>
                    ALPAKA_FN_HOST static auto createView(
                        TBuf & buf)
                    -> mem::view::ViewBasic<TDev, TElem, TDim, TSize>
                    {
                        return mem::view::ViewBasic<
                            TDev,
                            TElem,
                            TDim,
                            TSize>(
                                buf);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TBuf,
                        typename TExtents,
                        typename TOffsets>
                    ALPAKA_FN_HOST static auto createView(
                        TBuf const & buf,
                        TExtents const & extentsElements,
                        TOffsets const & relativeOffsetsElements)
                    -> mem::view::ViewBasic<typename std::add_const<TElem>::type, TDim, TDev, TSize>
                    {
                        return mem::view::ViewBasic<
                            TDev,
                            typename std::add_const<TElem>::type,
                            TDim,
                            TSize>(
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
                    ALPAKA_FN_HOST static auto createView(
                        TBuf & buf,
                        TExtents const & extentsElements,
                        TOffsets const & relativeOffsetsElements)
                    -> mem::view::ViewBasic<TDev, TElem, TDim, TSize>
                    {
                        return mem::view::ViewBasic<
                            TDev,
                            TElem,
                            TDim,
                            TSize>(
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
                    typename TDev,
                    typename TSize>
                struct GetBuf<
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        mem::view::ViewBasic<TDev, TElem, TDim, TSize> const & view)
                    -> typename mem::view::ViewBasic<TDev, TElem, TDim, TSize>::Buf const &
                    {
                        return view.m_Buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        mem::view::ViewBasic<TDev, TElem, TDim, TSize> & view)
                    -> typename mem::view::ViewBasic<TDev, TElem, TDim, TSize>::Buf &
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
                    typename TDev,
                    typename TSize>
                struct GetPtrNative<
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
                {
                private:
                    using IdxSequence = alpaka::core::detail::make_integer_sequence<std::size_t, TDim::value>;
                public:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewBasic<TDev, TElem, TDim, TSize> const & view)
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
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewBasic<TDev, TElem, TDim, TSize> & view)
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
                        std::size_t... TIndices>
                    ALPAKA_FN_HOST static auto pitchedOffsetBytes(
                        TView const & view,
                        TBuf const & buf,
                        alpaka::core::detail::integer_sequence<std::size_t, TIndices...> const &)
                    -> TSize
                    {
                        return
                            core::foldr(
                                std::plus<TSize>(),
                                pitchedOffsetBytesDim<TIndices>(view, buf)...);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        std::size_t Tidx,
                        typename TView,
                        typename TBuf>
                    ALPAKA_FN_HOST static auto pitchedOffsetBytesDim(
                        TView const & view,
                        TBuf const & buf)
                    -> TSize
                    {
                        return
                            offset::getOffset<Tidx>(view)
                            * view::getPitchBytes<Tidx + 1u>(buf);
                    }
                };

                //#############################################################################
                //! The ViewBasic pitch get trait specialization.
                //#############################################################################
                template<
                    typename TIdx,
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    TIdx,
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::view::ViewBasic<TDev, TElem, TDim, TSize> const & view)
                    -> TSize
                    {
                        return
                            view::getPitchBytes<TIdx::value>(
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
                typename TDev,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::view::ViewBasic<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    mem::view::ViewBasic<TDev, TElem, TDim, TSize> const & offset)
                -> TSize
                {
                    return offset.m_vOffsetsElements[TIdx::value];
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewBasic size type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct SizeType<
                mem::view::ViewBasic<TDev, TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
