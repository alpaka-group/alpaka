/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/Integral.hpp"
#include "alpaka/meta/NdLoop.hpp"

namespace alpaka
{
    class DevCpu;

    namespace detail
    {
        //! The CPU device ND memory fill task base.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskFillCpuBase
        {
            static_assert(TDim::value > 0);

            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            static_assert(std::is_trivially_copyable_v<Elem>, "Only trivially copyable types supported for fill");

            template<typename TViewFwd>
            TaskFillCpuBase(TViewFwd&& view, Elem const& value, TExtent const& extent)
                : m_value(value)
                , m_extent(getExtents(extent))
                , m_extentWidth(getExtents(extent).back())
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                , m_dstExtent(getExtents(view))
#endif
                , m_dstPitchBytes(getPitchesInBytes(view))
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))
            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).all());
                if constexpr(TDim::value > 1)
                    ALPAKA_ASSERT(
                        m_extentWidth * static_cast<ExtentSize>(sizeof(Elem)) <= m_dstPitchBytes[TDim::value - 2]);

                ALPAKA_ASSERT(reinterpret_cast<std::uintptr_t>(m_dstMemNative) % alignof(Elem) == 0);
            }

            Elem const m_value;
            Vec<TDim, ExtentSize> const m_extent;
            ExtentSize const m_extentWidth;
#if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
#endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            std::uint8_t* const m_dstMemNative;
        };

        //! Generic ND version memory fill task.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskFillCpu : public TaskFillCpuBase<TDim, TView, TExtent>
        {
            using TaskFillCpuBase<TDim, TView, TExtent>::TaskFillCpuBase;
            using typename TaskFillCpuBase<TDim, TView, TExtent>::Elem;
            using typename TaskFillCpuBase<TDim, TView, TExtent>::ExtentSize;

            ALPAKA_FN_HOST auto operator()() const -> void
            {
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    meta::ndLoopIncIdx(
                        this->m_extent,
                        [&](Vec<TDim, ExtentSize> const& idx)
                        {
                            std::uintptr_t offsetBytes
                                = static_cast<std::uintptr_t>((idx * this->m_dstPitchBytes).sum());
                            assert(offsetBytes % alignof(Elem) == 0);
                            Elem* elem = reinterpret_cast<Elem*>(
                                __builtin_assume_aligned(this->m_dstMemNative + offsetBytes, alignof(Elem)));

                            *elem = this->m_value;
                        });
                }
            }
        };

        // 0D version (scalar fill)
        template<typename TView, typename TExtent>
        struct TaskFillCpu<DimInt<0u>, TView, TExtent>
        {
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskFillCpu(TViewFwd&& view, Elem const& value, [[maybe_unused]] TExtent const& extent)
                : m_value(value)
                , m_dstMemNative(getPtrNative(view))
            {
                ALPAKA_ASSERT(getExtents(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtents(view).prod() == 1u);
                ALPAKA_ASSERT(reinterpret_cast<std::uintptr_t>(m_dstMemNative) % alignof(Elem) == 0);
            }

            ALPAKA_FN_HOST auto operator()() const noexcept -> void
            {
                *m_dstMemNative = m_value;
            }

            Elem const m_value;
            Elem* const m_dstMemNative;
        };
    } // namespace detail

    namespace trait
    {
        //! The memory fill task trait specialization for CPU devices.
        template<typename TDim>
        struct CreateTaskFill<TDim, DevCpu>
        {
            template<typename TExtent, typename TViewFwd>
            ALPAKA_FN_HOST static auto createTaskFill(
                TViewFwd&& view,
                alpaka::Elem<std::remove_reference_t<TViewFwd>> const& value,
                TExtent const& extent)
            {
                using TView = std::remove_reference_t<TViewFwd>;
                using Elem = alpaka::Elem<TView>;
                static_assert(
                    std::is_trivially_copyable_v<Elem>,
                    "Only trivially copyable types are supported for fill");

                return alpaka::detail::TaskFillCpu<TDim, TView, TExtent>{std::forward<TViewFwd>(view), value, extent};
            }
        };
    } // namespace trait

} // namespace alpaka
