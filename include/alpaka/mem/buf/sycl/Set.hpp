/* Copyright 2022 Jan Stephan, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Debug.hpp"
#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/mem/buf/sycl/Common.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/queue/QueueGenericSyclBlocking.hpp"
#include "alpaka/queue/QueueGenericSyclNonBlocking.hpp"
#include "alpaka/queue/Traits.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka
{

    namespace detail
    {
        //!  The SYCL ND memory set task base.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetSyclBase
        {
            using ExtentSize = Idx<TExtent>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskSetSyclBase(TViewFwd&& view, std::uint8_t const& byte, TExtent const& extent)
                : m_byte(byte)
                , m_extent(getExtentVec(extent))
                , m_extentWidthBytes(m_extent[TDim::value - 1u] * static_cast<ExtentSize>(sizeof(Elem)))
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
                , m_dstExtent(getExtentVec(view))
#    endif

                , m_dstPitchBytes(getPitchBytesVec(view))
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))

            {
                ALPAKA_ASSERT((castVec<DstSize>(m_extent) <= m_dstExtent).foldrAll(std::logical_or<bool>()));
                ALPAKA_ASSERT(m_extentWidthBytes <= m_dstPitchBytes[TDim::value - 1u]);
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " e: " << this->m_extent << " ewb: " << this->m_extentWidthBytes
                          << " de: " << this->m_dstExtent << " dptr: " << reinterpret_cast<void*>(this->m_dstMemNative)
                          << " dpitchb: " << this->m_dstPitchBytes << std::endl;
            }
#    endif

            std::uint8_t const m_byte;
            Vec<TDim, ExtentSize> const m_extent;
            ExtentSize const m_extentWidthBytes;
#    if(!defined(NDEBUG)) || (ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL)
            Vec<TDim, DstSize> const m_dstExtent;
#    endif
            Vec<TDim, DstSize> const m_dstPitchBytes;
            std::uint8_t* const m_dstMemNative;
            static constexpr auto is_sycl_task = true;
        };

        //! The SYCL device ND memory set task.
        template<typename TDim, typename TView, typename TExtent>
        struct TaskSetSycl : public TaskSetSyclBase<TDim, TView, TExtent>
        {
            using DimMin1 = DimInt<TDim::value - 1u>;
            using typename TaskSetSyclBase<TDim, TView, TExtent>::ExtentSize;
            using typename TaskSetSyclBase<TDim, TView, TExtent>::DstSize;

            using TaskSetSyclBase<TDim, TView, TExtent>::TaskSetSyclBase;

            auto operator()(sycl::handler& cgh) const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#    endif
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    cgh.memset(
                        reinterpret_cast<void*>(this->m_dstMemNative),
                        this->m_byte,
                        static_cast<std::size_t>(this->m_extentWidthBytes * this->m_extent.prod()));
                }
            }
        };

        //! The 1D SYCL memory set task.
        template<typename TView, typename TExtent>
        struct TaskSetSycl<DimInt<1u>, TView, TExtent> : public TaskSetSyclBase<DimInt<1u>, TView, TExtent>
        {
            using TaskSetSyclBase<DimInt<1u>, TView, TExtent>::TaskSetSyclBase;

            auto operator()(sycl::handler& cgh) const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                this->printDebug();
#    endif
                if(static_cast<std::size_t>(this->m_extent.prod()) != 0u)
                {
                    cgh.memset(
                        reinterpret_cast<void*>(this->m_dstMemNative),
                        this->m_byte,
                        static_cast<std::size_t>(this->m_extentWidthBytes));
                }
            }
        };

        //! The SYCL device scalar memory set task.
        template<typename TView, typename TExtent>
        struct TaskSetSycl<DimInt<0u>, TView, TExtent>
        {
            using ExtentSize = Idx<TExtent>;
            using Scalar = Vec<DimInt<0u>, ExtentSize>;
            using DstSize = Idx<TView>;
            using Elem = alpaka::Elem<TView>;

            template<typename TViewFwd>
            TaskSetSycl(TViewFwd&& view, std::uint8_t const& byte, [[maybe_unused]] TExtent const& extent)
                : m_byte(byte)
                , m_dstMemNative(reinterpret_cast<std::uint8_t*>(getPtrNative(view)))
            {
                // all zero-sized extents are equivalent
                ALPAKA_ASSERT(getExtentVec(extent).prod() == 1u);
                ALPAKA_ASSERT(getExtentVec(view).prod() == 1u);
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            ALPAKA_FN_HOST auto printDebug() const -> void
            {
                std::cout << __func__ << " e: " << Scalar() << " ewb: " << sizeof(Elem) << " de: " << Scalar()
                          << " dptr: " << reinterpret_cast<void*>(m_dstMemNative) << " dpitchb: " << Scalar()
                          << std::endl;
            }
#    endif

            auto operator()(sycl::handler& cgh) const -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                printDebug();
#    endif
                cgh.memset(reinterpret_cast<void*>(m_dstMemNative), m_byte, sizeof(Elem));
            }

            std::uint8_t const m_byte;
            std::uint8_t* const m_dstMemNative;
            static constexpr auto is_sycl_task = true;
        };

    } // namespace detail


    namespace trait
    {
        //! The SYCL device memory set trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemset<TDim, DevGenericSycl<TPltf>>
        {
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView& view, std::uint8_t const& byte, TExtent const& extent)
                -> detail::TaskSetSycl<TDim, TView, TExtent>
            {
                return detail::TaskSetSycl<TDim, TView, TExtent>(view, byte, extent);
            }
        };

    } // namespace trait

} // namespace alpaka
#endif
