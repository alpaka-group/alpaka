/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"
#include "alpaka/mem/buf/uniformCudaHip/BufUniformCudaHipRt.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::trait
{
    //! The BufUniformCudaHipRt device type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct DevType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = DevUniformCudaHipRt<TApi>;
    };

    //! The BufUniformCudaHipRt device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetDev<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> DevUniformCudaHipRt<TApi>
        {
            return GetDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>::getDev(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufUniformCudaHipRt dimension getter trait.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct DimType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The BufUniformCudaHipRt memory element type get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct ElemType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TElem;
    };

    //! The BufUniformCudaHipRt width get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetExtents<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
        {
            return GetExtents<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>{}(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufUniformCudaHipRt native pointer get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrNative<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return GetPtrNative<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>::getPtrNative(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>& buf) -> TElem*
        {
            // cast away the TElem's constness from the ConstBuf's return type
            return const_cast<TElem*>(GetPtrNative<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>::getPtrNative(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf}));
        }
    };

    //! The BufUniformCudaHipRt pointer on device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf,
            DevUniformCudaHipRt<TApi> const& /*dev*/) -> TElem const*
        {
            return GetPtrDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>::getPtrDev(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf});
        }

        ALPAKA_FN_HOST static auto getPtrDev(
            BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>& buf,
            DevUniformCudaHipRt<TApi> const& /*dev*/) -> TElem*
        {
            // cast away the TElem's constness from the ConstBuf's return type
            return const_cast<TElem*>(
                GetPtrDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>::getPtrDev(
                    ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf}));
        }
    };

    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPitchesInBytes<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf) const
            -> Vec<TDim, TIdx>
        {
            return GetPitchesInBytes<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>{}(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufUniformCudaHipRt offset get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetOffsets<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf) const
            -> Vec<TDim, TIdx>
        {
            return GetOffsets<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>{}(
                ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>{buf});
        }
    };

    //! The BufUniformCudaHipRt idx type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct IdxType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
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

    //! The MakeConstBuf trait for CUDA/HIP buffers.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct MakeConstBuf<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>
        {
            return ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>(buf);
        }
    };
} // namespace alpaka::trait

#endif
