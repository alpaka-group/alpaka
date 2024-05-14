/* Copyright 2023 Benjamin Worpitz, Matthias Werner, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber,
 *                Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/mem/Visibility.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/meta/TypeListOps.hpp"
#include "alpaka/platform/PlatformCpu.hpp"
#include "alpaka/platform/PlatformCpuSycl.hpp"
#include "alpaka/platform/PlatformCudaRt.hpp"
#include "alpaka/platform/PlatformFpgaSyclIntel.hpp"
#include "alpaka/platform/PlatformGenericSycl.hpp"
#include "alpaka/platform/PlatformGpuSyclIntel.hpp"
#include "alpaka/platform/PlatformHipRt.hpp"
#include "alpaka/vec/Vec.hpp"

#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The memory view to wrap plain pointers.
    template<
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        typename TMemVisibility =
            typename alpaka::meta::toTuple<typename alpaka::trait::MemVisibility<alpaka::Platform<TDev>>::type>::type>
    struct ViewPlainPtr final : internal::ViewAccessOps<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
    {
        static_assert(!std::is_const_v<TIdx>, "The idx type of the view can not be const!");

        template<typename TExtent>
        ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, TDev dev, TExtent const& extent = TExtent())
            : ViewPlainPtr(pMem, std::move(dev), extent, detail::calculatePitchesFromExtents<TElem>(extent))
        {
        }

        template<typename TExtent, typename TPitch>
        ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, TDev dev, TExtent const& extent, TPitch pitchBytes)
            : m_pMem(pMem)
            , m_dev(std::move(dev))
            , m_extentElements(extent)
            , m_pitchBytes(static_cast<Vec<TDim, TIdx>>(pitchBytes))
        {
        }

        TElem* m_pMem;
        TDev m_dev;
        Vec<TDim, TIdx> m_extentElements;
        Vec<TDim, TIdx> m_pitchBytes;
    };

    // Trait specializations for ViewPlainPtr.
    namespace trait
    {
        //! The ViewPlainPtr device type trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct DevType<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            using type = alpaka::Dev<TDev>;
        };

        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct MemVisibility<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            using type = TMemVisibility;
        };

        //! The ViewPlainPtr device get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct GetDev<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            static auto getDev(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) -> alpaka::Dev<TDev>
            {
                return view.m_dev;
            }
        };

        //! The ViewPlainPtr dimension getter trait.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct DimType<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            using type = TDim;
        };

        //! The ViewPlainPtr memory element type get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct ElemType<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            using type = TElem;
        };
    } // namespace trait

    namespace trait
    {
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct GetExtents<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            ALPAKA_FN_HOST auto operator()(ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility> const& view) const
            {
                return view.m_extentElements;
            }
        };

        //! The ViewPlainPtr native pointer get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct GetPtrNative<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            static auto getPtrNative(ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility> const& view) -> TElem const*
            {
                return view.m_pMem;
            }

            static auto getPtrNative(ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>& view) -> TElem*
            {
                return view.m_pMem;
            }
        };

        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct GetPitchesInBytes<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            ALPAKA_FN_HOST auto operator()(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) const
            {
                return view.m_pitchBytes;
            }
        };

        //! The CPU device CreateViewPlainPtr trait specialization.
        template<>
        struct CreateViewPlainPtr<DevCpu>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(DevCpu const& dev, TElem* pMem, TExtent const& extent, TPitch pitch)
            {
                return alpaka::ViewPlainPtr<DevCpu, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                    pMem,
                    dev,
                    extent,
                    pitch);
            }
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        //! The CUDA/HIP RT device CreateViewPlainPtr trait specialization.
        template<typename TApi>
        struct CreateViewPlainPtr<DevUniformCudaHipRt<TApi>>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(
                DevUniformCudaHipRt<TApi> const& dev,
                TElem* pMem,
                TExtent const& extent,
                TPitch pitch)
            {
                return alpaka::
                    ViewPlainPtr<DevUniformCudaHipRt<TApi>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                        pMem,
                        dev,
                        extent,
                        pitch);
            }
        };
#endif

#if defined(ALPAKA_ACC_SYCL_ENABLED)
        //! The SYCL device CreateViewPlainPtr trait specialization.
        template<typename TPlatform>
        struct CreateViewPlainPtr<DevGenericSycl<TPlatform>>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(
                DevGenericSycl<TPlatform> const& dev,
                TElem* pMem,
                TExtent const& extent,
                TPitch pitch)
            {
                return alpaka::
                    ViewPlainPtr<DevGenericSycl<TPlatform>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                        pMem,
                        dev,
                        extent,
                        pitch);
            }
        };
#endif
        //! The ViewPlainPtr offset get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct GetOffsets<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            ALPAKA_FN_HOST auto operator()(ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility> const&) const
                -> Vec<TDim, TIdx>
            {
                return Vec<TDim, TIdx>::zeros();
            }
        };

        //! The ViewPlainPtr idx type trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TMemVisibility>
        struct IdxType<ViewPlainPtr<TDev, TElem, TDim, TIdx, TMemVisibility>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka
