/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/mem/view/ViewPlainPtr.hpp"
#include "alpaka/offset/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <type_traits>
#include <utility>

namespace alpaka
{
    //! A sub-view to a view.
    template<typename TDev, typename TElem, typename TDim, typename TIdx>
    class ViewSubView : public internal::ViewAccessOps<ViewSubView<TDev, TElem, TDim, TIdx>>
    {
        static_assert(!std::is_const_v<TIdx>, "The idx type of the view can not be const!");

        using Dev = alpaka::Dev<TDev>;

    public:
        //! Constructor.
        //! \param view The view this view is a sub-view of.
        //! \param extentElements The extent in elements.
        //! \param relativeOffsetsElements The offsets in elements.
        template<typename TView, typename TOffsets, typename TExtent>
        ViewSubView(
            TView const& view,
            TExtent const& extentElements,
            TOffsets const& relativeOffsetsElements = TOffsets())
            : m_viewParentView(getPtrNative(view), getDev(view), getExtents(view), getPitchesInBytes(view))
            , m_extentElements(getExtents(extentElements))
            , m_offsetsElements(getOffsetVec(relativeOffsetsElements))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            static_assert(
                std::is_same_v<Dev, alpaka::Dev<TView>>,
                "The dev type of TView and the Dev template parameter have to be identical!");

            static_assert(
                std::is_same_v<TIdx, Idx<TView>>,
                "The idx type of TView and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same_v<TIdx, Idx<TOffsets>>,
                "The idx type of TOffsets and the TIdx template parameter have to be identical!");

            static_assert(
                std::is_same_v<TDim, Dim<TView>>,
                "The dim type of TView and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same_v<TDim, Dim<TExtent>>,
                "The dim type of TExtent and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same_v<TDim, Dim<TOffsets>>,
                "The dim type of TOffsets and the TDim template parameter have to be identical!");

            ALPAKA_ASSERT(
                ((m_offsetsElements + m_extentElements) <= getExtents(view)).foldrAll(std::logical_and<bool>(), true));
        }
        //! Constructor.
        //! \param view The view this view is a sub-view of.
        //! \param extentElements The extent in elements.
        //! \param relativeOffsetsElements The offsets in elements.
        template<typename TView, typename TOffsets, typename TExtent>
        ViewSubView(TView& view, TExtent const& extentElements, TOffsets const& relativeOffsetsElements = TOffsets())
            : m_viewParentView(getPtrNative(view), getDev(view), getExtents(view), getPitchesInBytes(view))
            , m_extentElements(getExtents(extentElements))
            , m_offsetsElements(getOffsetVec(relativeOffsetsElements))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            static_assert(
                std::is_same_v<Dev, alpaka::Dev<TView>>,
                "The dev type of TView and the Dev template parameter have to be identical!");

            static_assert(
                std::is_same_v<TIdx, Idx<TView>>,
                "The idx type of TView and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
            static_assert(
                std::is_same_v<TIdx, Idx<TOffsets>>,
                "The idx type of TOffsets and the TIdx template parameter have to be identical!");

            static_assert(
                std::is_same_v<TDim, Dim<TView>>,
                "The dim type of TView and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same_v<TDim, Dim<TExtent>>,
                "The dim type of TExtent and the TDim template parameter have to be identical!");
            static_assert(
                std::is_same_v<TDim, Dim<TOffsets>>,
                "The dim type of TOffsets and the TDim template parameter have to be identical!");

            ALPAKA_ASSERT(
                ((m_offsetsElements + m_extentElements) <= getExtents(view)).foldrAll(std::logical_and<bool>(), true));
        }

        //! \param view The view this view is a sub-view of.
        template<typename TView>
        explicit ViewSubView(TView const& view) : ViewSubView(view, view, Vec<TDim, TIdx>::all(0))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;
        }

        //! \param view The view this view is a sub-view of.
        template<typename TView>
        explicit ViewSubView(TView& view) : ViewSubView(view, view, Vec<TDim, TIdx>::all(0))
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;
        }

    public:
        ViewPlainPtr<Dev, TElem, TDim, TIdx> m_viewParentView; // This wraps the parent view.
        Vec<TDim, TIdx> m_extentElements; // The extent of this view.
        Vec<TDim, TIdx> m_offsetsElements; // The offset relative to the parent view.
    };

    // Trait specializations for ViewSubView.
    namespace trait
    {
        //! The ViewSubView device type trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct DevType<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            using type = alpaka::Dev<TDev>;
        };

        //! The ViewSubView device get trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct GetDev<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(ViewSubView<TDev, TElem, TDim, TIdx> const& view) -> alpaka::Dev<TDev>
            {
                return alpaka::getDev(view.m_viewParentView);
            }
        };

        //! The ViewSubView dimension getter trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct DimType<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The ViewSubView memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct ElemType<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            using type = TElem;
        };

        //! The ViewSubView width get trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct GetExtents<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST auto operator()(ViewSubView<TDev, TElem, TDim, TIdx> const& view) const
            {
                return view.m_extentElements;
            }
        };

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'std::uint8_t*' to 'TElem*' increases required alignment of target type"
#endif
        //! The ViewSubView native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct GetPtrNative<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
        private:
            using IdxSequence = std::make_index_sequence<TDim::value>;

        public:
            ALPAKA_FN_HOST static auto getPtrNative(ViewSubView<TDev, TElem, TDim, TIdx> const& view) -> TElem const*
            {
                // \TODO: pre-calculate this pointer for faster execution.
                return reinterpret_cast<TElem const*>(
                    reinterpret_cast<std::uint8_t const*>(alpaka::getPtrNative(view.m_viewParentView))
                    + pitchedOffsetBytes(view, IdxSequence()));
            }
            ALPAKA_FN_HOST static auto getPtrNative(ViewSubView<TDev, TElem, TDim, TIdx>& view) -> TElem*
            {
                // \TODO: pre-calculate this pointer for faster execution.
                return reinterpret_cast<TElem*>(
                    reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(view.m_viewParentView))
                    + pitchedOffsetBytes(view, IdxSequence()));
            }

        private:
            //! For a 3D vector this calculates:
            //!
            //! getOffsets(view)[0] * getPitchBytes<1u>(view)
            //! + getOffsets(view)[1] * getPitchBytes<2u>(view)
            //! + getOffsets(view)[2] * getPitchBytes<3u>(view)
            //! while getPitchBytes<3u>(view) is equivalent to sizeof(TElem)
            template<typename TView, std::size_t... TIndices>
            ALPAKA_FN_HOST static auto pitchedOffsetBytes(TView const& view, std::index_sequence<TIndices...> const&)
                -> TIdx
            {
                auto const offsets = getOffsets(view);
                auto const pitches = getPitchesInBytes(view);
                return (
                    (offsets[TIndices]
                     * (TIndices + 1 < Dim<TView>::value ? pitches[TIndices + 1]
                                                         : static_cast<TIdx>(sizeof(Elem<TView>))))
                    + ... + TIdx{0}); // FIXME: see comment above
            }
        };
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

        //! The ViewSubView pitch get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetPitchesInBytes<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST auto operator()(ViewSubView<TDev, TElem, TDim, TIdx> const& view) const
            {
                return alpaka::getPitchesInBytes(view.m_viewParentView);
            }
        };

        //! The ViewSubView x offset get trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct GetOffsets<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST auto operator()(ViewSubView<TDev, TElem, TDim, TIdx> const& offset)
            {
                return offset.m_offsetsElements;
            }
        };

        //! The ViewSubView idx type trait specialization.
        template<typename TElem, typename TDim, typename TDev, typename TIdx>
        struct IdxType<ViewSubView<TDev, TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The CPU device CreateSubView trait default implementation
        template<typename TDev, typename TSfinae>
        struct CreateSubView
        {
            template<typename TView, typename TExtent, typename TOffsets>
            static auto createSubView(
                TView& view,
                TExtent const& extentElements,
                TOffsets const& relativeOffsetsElements)
            {
                using Dim = alpaka::Dim<TExtent>;
                using Idx = alpaka::Idx<TExtent>;
                using Elem = typename trait::ElemType<TView>::type;
                return ViewSubView<TDev, Elem, Dim, Idx>(view, extentElements, relativeOffsetsElements);
            }
        };
    } // namespace trait
} // namespace alpaka
