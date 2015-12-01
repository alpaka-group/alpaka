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
#include <alpaka/extent/Traits.hpp>                 // mem::view::getXXX
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/offset/Traits.hpp>                 // traits::getOffsetX
#include <alpaka/size/Traits.hpp>                   // size::traits::SizeType

#include <alpaka/mem/view/ViewPlainPtr.hpp>         // ViewPlainPtr
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
            class ViewSubView
            {
            public:
                using Dev = TDev;
                using Elem = TElem;
                using Dim = TDim;
                using Buf = mem::buf::ViewPlainPtr<TDev, TElem, TDim, TSize>;
                // If the value type is const, we store a const buffer.
                //using BufC = detail::MimicConst<TElem, Buf>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                ViewSubView(
                    TBuf const & buf) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(buf)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(buf))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf>
                ViewSubView(
                    TBuf & buf) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(buf)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(buf))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TSize, size::Size<TBuf>>::value,
                        "The size type of TBuf and the TSize template parameter have to be identical!");
                }

                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //! \param extentElements The extent in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TOffsets,
                    typename TExtent>
                ViewSubView(
                    TBuf const & buf,
                    TExtent const & extentElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extentElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(relativeOffsetsElements) + offset::getOffsetsVecEnd<TDim>(buf))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TBuf>>::value,
                        "The size type of TBuf and the TSize template parameter have to be identical!");

                    assert(extent::getWidth(relativeOffsetsElements) <= extent::getWidth(buf));
                    assert(extent::getHeight(relativeOffsetsElements) <= extent::getHeight(buf));
                    assert(extent::getDepth(relativeOffsetsElements) <= extent::getDepth(buf));
                    assert((offset::getOffsetX(relativeOffsetsElements)+offset::getOffsetX(buf)+extent::getWidth(extentElements)) <= extent::getWidth(buf));
                    assert((offset::getOffsetY(relativeOffsetsElements)+offset::getOffsetY(buf)+extent::getHeight(extentElements)) <= extent::getHeight(buf));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+offset::getOffsetZ(buf)+extent::getDepth(extentElements)) <= extent::getDepth(buf));
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param buf This can be either a memory buffer or a memory view.
                //! \param extentElements The extent in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TBuf,
                    typename TOffsets,
                    typename TExtent>
                ViewSubView(
                    TBuf & buf,
                    TExtent const & extentElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::view::getPtrNative(buf),
                            dev::getDev(buf),
                            extent::getExtentVecEnd<TDim>(buf),
                            mem::view::getPitchBytes<TDim::value - 1u>(buf)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extentElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(relativeOffsetsElements) + offset::getOffsetsVecEnd<TDim>(buf))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TBuf>>::value,
                        "The size type of TBuf and the TSize template parameter have to be identical!");

                    assert(extent::getWidth(relativeOffsetsElements) <= extent::getWidth(buf));
                    assert(extent::getHeight(relativeOffsetsElements) <= extent::getHeight(buf));
                    assert(extent::getDepth(relativeOffsetsElements) <= extent::getDepth(buf));
                    assert((offset::getOffsetX(relativeOffsetsElements)+offset::getOffsetX(buf)+extent::getWidth(extentElements)) <= extent::getWidth(buf));
                    assert((offset::getOffsetY(relativeOffsetsElements)+offset::getOffsetY(buf)+extent::getHeight(extentElements)) <= extent::getHeight(buf));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+offset::getOffsetZ(buf)+extent::getDepth(extentElements)) <= extent::getDepth(buf));
                }

            public:
                Buf m_Buf;
                Vec<TDim, TSize> m_extentElements;
                Vec<TDim, TSize> m_vOffsetsElements;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for ViewSubView.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct DevType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The ViewSubView device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct GetDev<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
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
            //! The ViewSubView dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct DimType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct ElemType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewSubView width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & extent)
                -> TSize
                {
                    return extent.m_extentElements[TIdx::value];
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
                //! The ViewSubView buf trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev,
                    typename TSize>
                struct GetBuf<
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
                    -> typename mem::view::ViewSubView<TDev, TElem, TDim, TSize>::Buf const &
                    {
                        return view.m_Buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> & view)
                    -> typename mem::view::ViewSubView<TDev, TElem, TDim, TSize>::Buf &
                    {
                        return view.m_Buf;
                    }
                };

                //#############################################################################
                //! The ViewSubView native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim,
                    typename TDev,
                    typename TSize>
                struct GetPtrNative<
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
                {
                private:
                    using IdxSequence = alpaka::core::detail::make_integer_sequence<std::size_t, TDim::value>;
                public:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
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
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> & view)
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
                            * mem::view::getPitchBytes<Tidx + 1u>(buf);
                    }
                };

                //#############################################################################
                //! The ViewSubView pitch get trait specialization.
                //#############################################################################
                template<
                    typename TIdx,
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    TIdx,
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
                    -> TSize
                    {
                        return
                            mem::view::getPitchBytes<TIdx::value>(
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
            //! The ViewSubView x offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & offset)
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
            //! The ViewSubView size type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim,
                typename TDev,
                typename TSize>
            struct SizeType<
                mem::view::ViewSubView<TDev, TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
