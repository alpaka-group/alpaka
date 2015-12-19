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
                using Buf = mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>;
                // If the value type is const, we store a const buffer.
                //using BufC = detail::MimicConst<TElem, Buf>;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param view This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                ViewSubView(
                    TView const & view) :
                        m_Buf(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytes<TDim::value - 1u>(view)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(view))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param view This can be either a memory buffer or a memory view.
                //-----------------------------------------------------------------------------
                template<
                    typename TView>
                ViewSubView(
                    TView & view) :
                        m_Buf(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytes<TDim::value - 1u>(view)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(view))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TSize, size::Size<TView>>::value,
                        "The size type of TView and the TSize template parameter have to be identical!");
                }

                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param view This can be either a memory buffer or a memory view.
                //! \param extentElements The extent in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TView,
                    typename TOffsets,
                    typename TExtent>
                ViewSubView(
                    TView const & view,
                    TExtent const & extentElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytes<TDim::value - 1u>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extentElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(relativeOffsetsElements) + offset::getOffsetsVecEnd<TDim>(view))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TView>>::value,
                        "The size type of TView and the TSize template parameter have to be identical!");

                    assert(extent::getWidth(relativeOffsetsElements) <= extent::getWidth(view));
                    assert(extent::getHeight(relativeOffsetsElements) <= extent::getHeight(view));
                    assert(extent::getDepth(relativeOffsetsElements) <= extent::getDepth(view));
                    assert((offset::getOffsetX(relativeOffsetsElements)+offset::getOffsetX(view)+extent::getWidth(extentElements)) <= extent::getWidth(view));
                    assert((offset::getOffsetY(relativeOffsetsElements)+offset::getOffsetY(view)+extent::getHeight(extentElements)) <= extent::getHeight(view));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+offset::getOffsetZ(view)+extent::getDepth(extentElements)) <= extent::getDepth(view));
                }
                //-----------------------------------------------------------------------------
                //! Constructor.
                //! \param view This can be either a memory buffer or a memory view.
                //! \param extentElements The extent in elements.
                //! \param relativeOffsetsElements The offsets in elements.
                //-----------------------------------------------------------------------------
                template<
                    typename TView,
                    typename TOffsets,
                    typename TExtent>
                ViewSubView(
                    TView & view,
                    TExtent const & extentElements,
                    TOffsets const & relativeOffsetsElements = TOffsets()) :
                        m_Buf(
                            mem::view::getPtrNative(view),
                            dev::getDev(view),
                            extent::getExtentVecEnd<TDim>(view),
                            mem::view::getPitchBytes<TDim::value - 1u>(view)),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extentElements)),
                        m_vOffsetsElements(offset::getOffsetsVecEnd<TDim>(relativeOffsetsElements) + offset::getOffsetsVecEnd<TDim>(view))
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    static_assert(
                        std::is_same<TDim, dim::Dim<TExtent>>::value,
                        "The buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                    static_assert(
                        std::is_same<TSize, size::Size<TView>>::value,
                        "The size type of TView and the TSize template parameter have to be identical!");

                    assert(extent::getWidth(relativeOffsetsElements) <= extent::getWidth(view));
                    assert(extent::getHeight(relativeOffsetsElements) <= extent::getHeight(view));
                    assert(extent::getDepth(relativeOffsetsElements) <= extent::getDepth(view));
                    assert((offset::getOffsetX(relativeOffsetsElements)+offset::getOffsetX(view)+extent::getWidth(extentElements)) <= extent::getWidth(view));
                    assert((offset::getOffsetY(relativeOffsetsElements)+offset::getOffsetY(view)+extent::getHeight(extentElements)) <= extent::getHeight(view));
                    assert((offset::getOffsetZ(relativeOffsetsElements)+offset::getOffsetZ(view)+extent::getDepth(extentElements)) <= extent::getDepth(view));
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
                            view);
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
                    using IdxSequence = meta::MakeIntegerSequence<std::size_t, TDim::value>;
                public:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> const & view)
                    -> TElem const *
                    {
                        // \TODO: pre-calculate this pointer for faster execution.
                        return
                            reinterpret_cast<TElem const *>(
                                reinterpret_cast<std::uint8_t const *>(mem::view::getPtrNative(view))
                                + pitchedOffsetBytes(view, view, IdxSequence()));
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::view::ViewSubView<TDev, TElem, TDim, TSize> & view)
                    -> TElem *
                    {
                        // \TODO: pre-calculate this pointer for faster execution.
                        return
                            reinterpret_cast<TElem *>(
                                reinterpret_cast<std::uint8_t *>(mem::view::getPtrNative(view))
                                + pitchedOffsetBytes(view, IdxSequence()));
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TView,
                        std::size_t... TIndices>
                    ALPAKA_FN_HOST static auto pitchedOffsetBytes(
                        TView const & view,
                        meta::IntegerSequence<std::size_t, TIndices...> const &)
                    -> TSize
                    {
                        return
                            meta::foldr(
                                std::plus<TSize>(),
                                pitchedOffsetBytesDim<TIndices>(view)...);
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        std::size_t Tidx,
                        typename TView>
                    ALPAKA_FN_HOST static auto pitchedOffsetBytesDim(
                        TView const & view)
                    -> TSize
                    {
                        return
                            offset::getOffset<Tidx>(view)
                            * mem::view::getPitchBytes<Tidx + 1u>(view);
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
                                view);
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
