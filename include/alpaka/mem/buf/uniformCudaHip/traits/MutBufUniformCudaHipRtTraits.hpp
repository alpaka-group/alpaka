/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/CpuBuf.hpp"
#include "alpaka/mem/buf/uniformCudaHip/MutBufUniformCudaHipRt.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::trait
{
    //! The MutBufUniformCudaHipRt device type trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DevType<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        using type = TDev;
    };

    //! The MutBufUniformCudaHipRt device get trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetDev<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx> const& buf) -> TDev
        {
            return GetDev<TBuf<TApi, TElem, TDim, TIdx>>::getDev(TBuf<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutBufUniformCudaHipRt dimension getter trait.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct DimType<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The MutBufUniformCudaHipRt memory element type get trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct ElemType<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The MutBufUniformCudaHipRt width get trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetExtents<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx> const& buf)
        {
            return GetExtents<TBuf<TApi, TElem, TDim, TIdx>>{}(TBuf<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutBufUniformCudaHipRt native pointer get trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetPtrNative<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx> const& buf) -> TElem const*
        {
            return GetPtrNative<TBuf<TApi, TElem, TDim, TIdx>>::getPtrNative(TBuf<TApi, TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>& buf) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrNative<TBuf<TApi, TElem, TDim, TIdx>>::getPtrNative(TBuf<TApi, TElem, TDim, TIdx>{buf}));
        }
    };

    //! The MutBufUniformCudaHipRt pointer on device get trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetPtrDev<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>, TDev>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx> const& buf,
            TDev const& /*dev*/) -> TElem const*
        {
            return GetPtrDev<TBuf<TApi, TElem, TDim, TIdx>, TDev>::getPtrDev(TBuf<TApi, TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>& buf,
            TDev const& /*dev*/) -> TElem*
        {
            // cast away the TElem's constness from the TBuf's return type
            return const_cast<TElem*>(
                GetPtrDev<TBuf<TApi, TElem, TDim, TIdx>, TDev>::getPtrDev(TBuf<TApi, TElem, TDim, TIdx>{buf}));
        }
    };

    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetPitchesInBytes<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx> const& buf) const -> Vec<TDim, TIdx>
        {
            return GetPitchesInBytes<TBuf<TApi, TElem, TDim, TIdx>>{}(TBuf<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutBufUniformCudaHipRt offset get trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct GetOffsets<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(
            MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx> const& buf) const -> Vec<TDim, TIdx>
        {
            return GetOffsets<TBuf<TApi, TElem, TDim, TIdx>>{}(TBuf<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The MutBufUniformCudaHipRt idx type trait specialization.
    template<
        template<typename, typename, typename, typename>
        class TBuf,
        typename TBufImpl,
        typename TApi,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    struct IdxType<MutBufUniformCudaHipRt<TBuf, TBufImpl, TApi, TDev, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on CUDA/HIP device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevUniformCudaHipRt<TApi> const&)
            -> TElem const*
        {
            // TODO: Check if the memory is mapped at all!
            TElem* pDev(nullptr);

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostGetDevicePointer(
                &pDev,
                const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                0));

            return pDev;
        }

        ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevUniformCudaHipRt<TApi> const&)
            -> TElem*
        {
            // TODO: Check if the memory is mapped at all!
            TElem* pDev(nullptr);

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostGetDevicePointer(&pDev, getPtrNative(buf), 0));

            return pDev;
        }
    };
} // namespace alpaka::trait

#endif
