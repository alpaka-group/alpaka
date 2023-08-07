/* Copyright 2023 Jan Stephan, Bernhard Manfred Gruber, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Debug.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/elem/Traits.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/buf/sycl/Common.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/queue/QueueGenericSyclBlocking.hpp"
#include "alpaka/queue/QueueGenericSyclNonBlocking.hpp"

#include <memory>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka::detail
{
    //!  The SYCL device memory copy task base.
    template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySyclBase
    {
        static_assert(
            std::is_same_v<std::remove_const_t<alpaka::Elem<TViewSrc>>, std::remove_const_t<alpaka::Elem<TViewDst>>>,
            "The source and the destination view are required to have the same element type!");
        using ExtentSize = Idx<TExtent>;
        using DstSize = Idx<TViewDst>;
        using SrcSize = Idx<TViewSrc>;
        using Elem = alpaka::Elem<TViewSrc>;

        template<typename TViewFwd>
        TaskCopySyclBase(TViewFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
            : m_extent(getExtentVec(extent))
            , m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem)))
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            , m_dstExtent(getExtentVec(viewDst))
            , m_srcExtent(getExtentVec(viewSrc))
#    endif
            , m_dstPitchBytes(getPitchBytesVec(viewDst))
            , m_srcPitchBytes(getPitchBytesVec(viewSrc))
            , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(viewDst)))
            , m_srcMemNative(reinterpret_cast<std::uint8_t const*>(getPtrNative(viewSrc)))
        {
            if constexpr(TDim::value > 0)
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                ALPAKA_ASSERT((castVec<SrcSize>(m_extent) <= m_srcExtent).foldrAll(std::logical_or<bool>()));
            }
        }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
        auto printDebug() const -> void
        {
            std::cout << __func__ << " e: " << m_extent << " ewb: " << this->m_extentWidthBytes
                      << " de: " << m_dstExtent << " dptr: " << reinterpret_cast<void*>(m_dstMemNative)
                      << " se: " << m_srcExtent << " sptr: " << reinterpret_cast<void const*>(m_srcMemNative)
                      << std::endl;
        }
#    endif

        Vec<TDim, ExtentSize> const m_extent;
        ExtentSize const m_extentWidthBytes;
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
        Vec<TDim, DstSize> const m_dstExtent;
        Vec<TDim, SrcSize> const m_srcExtent;
#    endif

        Vec<TDim, DstSize> const m_dstPitchBytes;
        Vec<TDim, SrcSize> const m_srcPitchBytes;
        std::uint8_t* const m_dstMemNative;
        std::uint8_t const* const m_srcMemNative;
        static constexpr auto is_sycl_task = true;
    };

    //! The SYCL device ND memory copy task.
    template<typename TDim, typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySycl : public TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>
    {
        using DimMin1 = DimInt<TDim::value - 1u>;
        using typename TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::ExtentSize;
        using typename TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::DstSize;
        using typename TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::SrcSize;

        using TaskCopySyclBase<TDim, TViewDst, TViewSrc, TExtent>::TaskCopySyclBase;

        auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            this->printDebug();
#    endif
            // [z, y, x] -> [z, y] because all elements with the innermost x dimension are handled within one
            // iteration.
            Vec<DimMin1, ExtentSize> const extentWithoutInnermost(subVecBegin<DimMin1>(this->m_extent));
            // [z, y, x] -> [y, x] because the z pitch (the full size of the buffer) is not required.
            Vec<DimMin1, DstSize> const dstPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_dstPitchBytes));
            Vec<DimMin1, SrcSize> const srcPitchBytesWithoutOutmost(subVecEnd<DimMin1>(this->m_srcPitchBytes));

            // Record an event for each memcpy call
            std::vector<sycl::event> events;
            events.reserve(static_cast<std::size_t>(extentWithoutInnermost.prod()));

            if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
            {
                meta::ndLoopIncIdx(
                    extentWithoutInnermost,
                    [&](Vec<DimMin1, ExtentSize> const& idx)
                    {
                        events.push_back(queue.memcpy(
                            reinterpret_cast<void*>(
                                this->m_dstMemNative
                                + (castVec<DstSize>(idx) * dstPitchBytesWithoutOutmost)
                                      .foldrAll(std::plus<DstSize>())),
                            reinterpret_cast<void const*>(
                                this->m_srcMemNative
                                + (castVec<SrcSize>(idx) * srcPitchBytesWithoutOutmost)
                                      .foldrAll(std::plus<SrcSize>())),
                            static_cast<std::size_t>(this->m_extentWidthBytes),
                            requirements));
                    });
            }

            // Return an event that depends on all the events assciated to the memcpy calls
            return queue.ext_oneapi_submit_barrier(events);
        }
    };

    //! The SYCL device 1D memory copy task.
    template<typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySycl<DimInt<1u>, TViewDst, TViewSrc, TExtent>
        : TaskCopySyclBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>
    {
        using TaskCopySyclBase<DimInt<1u>, TViewDst, TViewSrc, TExtent>::TaskCopySyclBase;
        using Elem = alpaka::Elem<TViewSrc>;

        auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            this->printDebug();
#    endif
            if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
            {
                return queue.memcpy(
                    reinterpret_cast<void*>(this->m_dstMemNative),
                    reinterpret_cast<void const*>(this->m_srcMemNative),
                    sizeof(Elem) * static_cast<std::size_t>(this->m_extent.prod()),
                    requirements);
            }
            else
            {
                return queue.ext_oneapi_submit_barrier();
            }
        }
    };

    //! The scalar SYCL memory copy trait.
    template<typename TViewDst, typename TViewSrc, typename TExtent>
    struct TaskCopySycl<DimInt<0u>, TViewDst, TViewSrc, TExtent>
    {
        static_assert(
            std::is_same_v<std::remove_const_t<alpaka::Elem<TViewSrc>>, std::remove_const_t<alpaka::Elem<TViewDst>>>,
            "The source and the destination view are required to have the same element type!");

        using Elem = alpaka::Elem<TViewSrc>;

        template<typename TViewDstFwd>
        TaskCopySycl(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, [[maybe_unused]] TExtent const& extent)
            : m_dstMemNative(reinterpret_cast<void*>(getPtrNative(viewDst)))
            , m_srcMemNative(reinterpret_cast<void const*>(getPtrNative(viewSrc)))
        {
            // all zero-sized extents are equivalent
            ALPAKA_ASSERT(getExtentVec(extent).prod() == 1u);
            ALPAKA_ASSERT(getExtentVec(viewDst).prod() == 1u);
            ALPAKA_ASSERT(getExtentVec(viewSrc).prod() == 1u);
        }

        auto operator()(sycl::queue& queue, std::vector<sycl::event> const& requirements) const -> sycl::event
        {
            return queue.memcpy(m_dstMemNative, m_srcMemNative, sizeof(Elem), requirements);
        }

        void* m_dstMemNative;
        void const* m_srcMemNative;
        static constexpr auto is_sycl_task = true;
    };
} // namespace alpaka::detail

// Trait specializations for CreateTaskMemcpy.
namespace alpaka::trait
{
    //! The SYCL host-to-device memory copy trait specialization.
    template<typename TPlatform, typename TDim>
    struct CreateTaskMemcpy<TDim, DevGenericSycl<TPlatform>, DevCpu>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        static auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
            -> alpaka::detail::TaskCopySycl<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            return {std::forward<TViewDstFwd>(viewDst), viewSrc, extent};
        }
    };

    //! The SYCL device-to-host memory copy trait specialization.
    template<typename TPlatform, typename TDim>
    struct CreateTaskMemcpy<TDim, DevCpu, DevGenericSycl<TPlatform>>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        static auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
            -> alpaka::detail::TaskCopySycl<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            return {std::forward<TViewDstFwd>(viewDst), viewSrc, extent};
        }
    };

    //! The SYCL device-to-device memory copy trait specialization.
    template<typename TPlatformDst, typename TPlatformSrc, typename TDim>
    struct CreateTaskMemcpy<TDim, DevGenericSycl<TPlatformDst>, DevGenericSycl<TPlatformSrc>>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        static auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& extent)
            -> alpaka::detail::TaskCopySycl<TDim, std::remove_reference_t<TViewDstFwd>, TViewSrc, TExtent>
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            return {std::forward<TViewDstFwd>(viewDst), viewSrc, extent};
        }
    };
} // namespace alpaka::trait

#endif
