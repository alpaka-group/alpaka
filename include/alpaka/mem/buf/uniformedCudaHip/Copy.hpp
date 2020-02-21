/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/queue/QueueUniformedCudaHipRtBlocking.hpp>
#include <alpaka/queue/QueueUniformedCudaHipRtNonBlocking.hpp>

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevUniformedCudaHipRt.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/queue/QueueUniformedCudaHipRtNonBlocking.hpp>
#include <alpaka/queue/QueueUniformedCudaHipRtBlocking.hpp>

#include <alpaka/core/Assert.hpp>
// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    #include <alpaka/core/Cuda.hpp>
#else
    #include <alpaka/core/Hip.hpp>
#endif

#include <set>
#include <tuple>


namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace uniformedCudaHip
            {
                namespace detail
                {
                    //#############################################################################
                    //! The CUDA-HIP memory copy trait.
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformedCudaHip;

                    //#############################################################################
                    //! The 1D CUDA-HIP memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformedCudaHip<
                        dim::DimInt<1>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, std::remove_const_t<elem::Elem<TViewSrc>>>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyUniformedCudaHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            cudaMemcpyKind const & cudaMemcpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemcpyKind),
#else
                            hipMemcpyKind const & hipMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_hipMemCpyKind(hipMemCpyKind),
#endif
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                            ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ddev: " << m_iDstDevice
                                << " ew: " << m_extentWidth
                                << " ewb: " << m_extentWidthBytes
                                << " dw: " << m_dstWidth
                                << " dptr: " << m_dstMemNative
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sptr: " << m_srcMemNative
                                << std::endl;
                        }
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        cudaMemcpyKind m_cudaMemCpyKind;
#else
                        hipMemcpyKind m_hipMemCpyKind;
#endif
                        int m_iDstDevice;
                        int m_iSrcDevice;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_extentWidth;
                        Idx m_dstWidth;
                        Idx m_srcWidth;
#endif
                        Idx m_extentWidthBytes;
                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 2D CUDA-HIP memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformedCudaHip<
                        dim::DimInt<2>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, std::remove_const_t<elem::Elem<TViewSrc>>>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyUniformedCudaHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            cudaMemcpyKind const & cudaMemcpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_cudaMemCpyKind(cudaMemcpyKind),
#else
                            hipMemcpyKind const & hipMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_hipMemCpyKind(hipMemCpyKind),
#endif
                            m_iDstDevice(iDstDevice),
                            m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),      // required for 3D peer copy
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),      // required for 3D peer copy

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst))),    // required for 3D peer copy
                                m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc))),    // required for 3D peer copy

                                m_dstpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),

                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                            ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_dstpitchBytesX);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstpitchBytesX
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcpitchBytesX
                                << std::endl;
                        }
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        cudaMemcpyKind m_cudaMemCpyKind;
#else
                        hipMemcpyKind m_hipMemCpyKind;
#endif
                        int m_iDstDevice;
                        int m_iSrcDevice;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_extentWidth;
#endif
                        Idx m_extentWidthBytes;
                        Idx m_dstWidth;          // required for 3D peer copy
                        Idx m_srcWidth;          // required for 3D peer copy

                        Idx m_extentHeight;
                        Idx m_dstHeight;         // required for 3D peer copy
                        Idx m_srcHeight;         // required for 3D peer copy

                        Idx m_dstpitchBytesX;
                        Idx m_srcpitchBytesX;
                        Idx m_dstPitchBytesY;
                        Idx m_srcPitchBytesY;


                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };
                    //#############################################################################
                    //! The 3D CUDA-HIP memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyUniformedCudaHip<
                        dim::DimInt<3>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, std::remove_const_t<elem::Elem<TViewSrc>>>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyUniformedCudaHip(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            cudaMemcpyKind const & cudaMemcpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                            m_cudaMemCpyKind(cudaMemcpyKind),
#else
                            hipMemcpyKind const & hipMemCpyKind,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                            m_hipMemCpyKind(hipMemCpyKind),
#endif
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),

                                m_extentWidth(extent::getWidth(extent)),
                                m_extentWidthBytes(m_extentWidth * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),

                                m_extentHeight(extent::getHeight(extent)),
                                m_dstHeight(static_cast<Idx>(extent::getHeight(viewDst))),
                                m_srcHeight(static_cast<Idx>(extent::getHeight(viewSrc))),

                                m_extentDepth(extent::getDepth(extent)),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_dstDepth(static_cast<Idx>(extent::getDepth(viewDst))),
                                m_srcDepth(static_cast<Idx>(extent::getDepth(viewSrc))),
#endif
                                m_dstpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - 1u>(viewDst))),
                                m_srcpitchBytesX(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - 1u>(viewSrc))),
                                m_dstPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewDst>::value - (2u % dim::Dim<TViewDst>::value)>(viewDst))),
                                m_srcPitchBytesY(static_cast<Idx>(mem::view::getPitchBytes<dim::Dim<TViewSrc>::value - (2u % dim::Dim<TViewDst>::value)>(viewSrc))),


                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_dstHeight);
                            ALPAKA_ASSERT(m_extentDepth <= m_dstDepth);
                            ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
                            ALPAKA_ASSERT(m_extentHeight <= m_srcHeight);
                            ALPAKA_ASSERT(m_extentDepth <= m_srcDepth);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_dstpitchBytesX);
                            ALPAKA_ASSERT(m_extentWidthBytes <= m_srcpitchBytesX);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ew: " << m_extentWidth
                                << " eh: " << m_extentHeight
                                << " ed: " << m_extentDepth
                                << " ewb: " << m_extentWidthBytes
                                << " ddev: " << m_iDstDevice
                                << " dw: " << m_dstWidth
                                << " dh: " << m_dstHeight
                                << " dd: " << m_dstDepth
                                << " dptr: " << m_dstMemNative
                                << " dpitchb: " << m_dstpitchBytesX
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sh: " << m_srcHeight
                                << " sd: " << m_srcDepth
                                << " sptr: " << m_srcMemNative
                                << " spitchb: " << m_srcpitchBytesX
                                << std::endl;
                        }
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        cudaMemcpyKind m_cudaMemCpyKind;
#else
                        hipMemcpyKind m_hipMemCpyKind;
#endif
                        int m_iDstDevice;
                        int m_iSrcDevice;

                        Idx m_extentWidth;
                        Idx m_extentWidthBytes;
                        Idx m_dstWidth;
                        Idx m_srcWidth;

                        Idx m_extentHeight;
                        Idx m_dstHeight;
                        Idx m_srcHeight;

                        Idx m_extentDepth;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_dstDepth;
                        Idx m_srcDepth;
#endif
                        Idx m_dstpitchBytesX;
                        Idx m_srcpitchBytesX;
                        Idx m_dstPitchBytesY;
                        Idx m_srcPitchBytesY;

                        void * m_dstMemNative;
                        void const * m_srcMemNative;
                    };

                    //-----------------------------------------------------------------------------
                    //! Not being able to enable peer access does not prevent such device to device memory copies.
                    //! However, those copies may be slower because the memory is copied via the CPU.
                    inline auto enablePeerAccessIfPossible(
                        const int & devSrc,
                        const int & devDst)
                    -> void
                    {
                        ALPAKA_ASSERT(devSrc != devDst);

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
                        static std::set<std::pair<int, int>> alreadyCheckedPeerAccessDevices;
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                        auto const devicePair = std::make_pair(devSrc, devDst);

                        if(alreadyCheckedPeerAccessDevices.find(devicePair) == alreadyCheckedPeerAccessDevices.end())
                        {
                            alreadyCheckedPeerAccessDevices.insert(devicePair);

                            int canAccessPeer = 0;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            ALPAKA_CUDA_RT_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, devSrc, devDst));
#else
                            ALPAKA_HIP_RT_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, devSrc, devDst));
#endif
                            if(!canAccessPeer) {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << __func__
                                << " Direct peer access between given GPUs is not possible!"
                                << " src=" << devSrc
                                << " dst=" << devDst
                                << std::endl;
#endif
                                return;
                            }
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            ALPAKA_CUDA_RT_CHECK(cudaSetDevice(devSrc));
#else
                            ALPAKA_HIP_RT_CHECK(hipSetDevice(devSrc));
#endif
                            // NOTE: "until access is explicitly disabled using cudaDeviceDisablePeerAccess() or either device is reset using cudaDeviceReset()."
                            // We do not remove a device from the enabled device pairs on cudaDeviceReset.
                            // Note that access granted by this call is unidirectional and that in order to access memory on the current device from peerDevice, a separate symmetric call to cudaDeviceEnablePeerAccess() is required.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                            ALPAKA_CUDA_RT_CHECK(cudaDeviceEnablePeerAccess(devDst, 0));
#else
                            ALPAKA_HIP_RT_CHECK(hipDeviceEnablePeerAccess(devDst, 0));
#endif
                        }
                    }
                }
            }

            //-----------------------------------------------------------------------------
            // Trait specializations for CreateTaskCopy.
            namespace traits
            {
                //#############################################################################
                //! The CUDA-HIP to CPU memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevUniformedCudaHipRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewSrc).m_iDevice);

                        return
                            mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    cudaMemcpyDeviceToHost,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The CPU to CUDA-HIP memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevUniformedCudaHipRt,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const iDevice(
                            dev::getDev(viewDst).m_iDevice);

                        return
                            mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    cudaMemcpyHostToDevice,
                                    iDevice,
                                    iDevice);
                    }
                };
                //#############################################################################
                //! The CUDA-HIP to CUDA-HIP memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevUniformedCudaHipRt,
                    dev::DevUniformedCudaHipRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    cudaMemcpyDeviceToDevice,
                                    dev::getDev(viewDst).m_iDevice,
                                    dev::getDev(viewSrc).m_iDevice);
                    }
                };
            }
            namespace uniformedCudaHip
            {
                namespace detail
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST auto buildUniformedCudaHipMemcpy3DParms(
                        mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<3>, TViewDst, TViewSrc, TExtent> const & task)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    -> cudaMemcpy3DParms
#else
                    -> hipMemcpy3DParms
#endif
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        //auto const & dstHeight(task.m_dstHeight);
                        //auto const & srcHeight(task.m_srcHeight);

                        auto const & extentDepth(task.m_extentDepth);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA-HIP parameter structure.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        cudaMemcpy3DParms cudaMemCpy3DParms;
                        cudaMemCpy3DParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                        cudaMemCpy3DParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(srcWidth),
                                static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));
                        cudaMemCpy3DParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                        cudaMemCpy3DParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                static_cast<std::size_t>(dstWidth),
                                static_cast<std::size_t>(dstPitchBytesY / dstPitchBytesX));
                        cudaMemCpy3DParms.extent =
                            make_cudaExtent(
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                static_cast<std::size_t>(extentDepth));
                        cudaMemCpy3DParms.kind = task.m_cudaMemCpyKind;
                        return cudaMemCpy3DParms;

#else
                        hipMemcpy3DParms hipMemCpy3DParms;
                        hipMemCpy3DParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                        hipMemCpy3DParms.srcPos = make_hipPos(0, 0, 0);  // Optional. Offset in bytes.
                        hipMemCpy3DParms.srcPtr =
                            make_hipPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(srcWidth),
                                static_cast<std::size_t>(srcPitchBytesY/srcPitchBytesX));
                        hipMemCpy3DParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                        hipMemCpy3DParms.dstPos = make_hipPos(0, 0, 0);  // Optional. Offset in bytes.
                        hipMemCpy3DParms.dstPtr =
                            make_hipPitchedPtr(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                static_cast<std::size_t>(dstWidth),
                                static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        hipMemCpy3DParms.extent =
                            make_hipExtent(
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                static_cast<std::size_t>(extentDepth));
    #ifdef __HIP_PLATFORM_NVCC__
                        hipMemCpy3DParms.kind = hipMemcpyKindToCudaMemcpyKind(task.m_hipMemCpyKind);
    #else
                        hipMemCpy3DParms.kind = task.m_hipMemCpyKind;
    #endif
                        return hipMemCpy3DParms;

#endif
                    }
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    //-----------------------------------------------------------------------------
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms(
                        mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<2>, TViewDst, TViewSrc, TExtent> const & task)
                    -> cudaMemcpy3DPeerParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & iDstDev(task.m_iDstDevice);
                        auto const & iSrcDev(task.m_iSrcDevice);

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        //auto const & dstHeight(task.m_dstHeight);
                        //auto const & srcHeight(task.m_srcHeight);

                        auto const extentDepth(1u);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                        cudaMemCpy3DPeerParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                static_cast<std::size_t>(dstWidth),
                                static_cast<std::size_t>(dstPitchBytesY / dstPitchBytesX));
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                static_cast<std::size_t>(extentDepth));
                        cudaMemCpy3DPeerParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(srcWidth),
                                static_cast<std::size_t>(srcPitchBytesY / srcPitchBytesX));

                        return cudaMemCpy3DPeerParms;
                    }
                    //-----------------------------------------------------------------------------
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    ALPAKA_FN_HOST auto buildCudaMemcpy3DPeerParms(
                        mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<3>, TViewDst, TViewSrc, TExtent> const & task)
                    -> cudaMemcpy3DPeerParms
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto const & iDstDev(task.m_iDstDevice);
                        auto const & iSrcDev(task.m_iSrcDevice);

                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & dstWidth(task.m_dstWidth);
                        auto const & srcWidth(task.m_srcWidth);

                        auto const & extentHeight(task.m_extentHeight);
                        //auto const & dstHeight(task.m_dstHeight);
                        //auto const & srcHeight(task.m_srcHeight);

                        auto const & extentDepth(task.m_extentDepth);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);
                        auto const & dstPitchBytesY(task.m_dstPitchBytesY);
                        auto const & srcPitchBytesY(task.m_srcPitchBytesY);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);

                        // Fill CUDA parameter structure.
                        cudaMemcpy3DPeerParms cudaMemCpy3DPeerParms;
                        cudaMemCpy3DPeerParms.dstArray = nullptr;  // Either dstArray or dstPtr.
                        cudaMemCpy3DPeerParms.dstDevice = iDstDev;
                        cudaMemCpy3DPeerParms.dstPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.dstPtr =
                            make_cudaPitchedPtr(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                static_cast<std::size_t>(dstWidth),
                                static_cast<std::size_t>(dstPitchBytesY/dstPitchBytesX));
                        cudaMemCpy3DPeerParms.extent =
                            make_cudaExtent(
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                static_cast<std::size_t>(extentDepth));
                        cudaMemCpy3DPeerParms.srcArray = nullptr;  // Either srcArray or srcPtr.
                        cudaMemCpy3DPeerParms.srcDevice = iSrcDev;
                        cudaMemCpy3DPeerParms.srcPos = make_cudaPos(0, 0, 0);  // Optional. Offset in bytes.
                        cudaMemCpy3DPeerParms.srcPtr =
                            make_cudaPitchedPtr(
                                const_cast<void *>(srcNativePtr),
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(srcWidth),
                                static_cast<std::size_t>(srcPitchBytesY / srcPitchBytesX));

                        return cudaMemCpy3DPeerParms;
                    }
#endif
                }
            }
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA-HIP non-blocking device queue 1D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformedCudaHipRtNonBlocking,
                mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformedCudaHipRtNonBlocking & queue,
                    mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    if(task.m_extentWidthBytes == 0)
                    {
                        return;
                    }

                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    {
                        auto const & uniformedCudaHipMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyAsync(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                uniformedCudaHipMemCpyKind,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);

                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeerAsync(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes),
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
#else
                    {
                    auto const & hipMemCpyKind(task.m_hipMemCpyKind);

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpyAsync(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                uniformedCudaHipMemCpyKind,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpyPeerAsync(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes),
                                queue.m_spQueueImpl->m_HipCudaQueue));
                    }
#endif
                }
            };
            //#############################################################################
            //! The CUDA-HIP blocking device queue 1D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformedCudaHipRtBlocking,
                mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformedCudaHipRtBlocking & queue,
                    mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    if(task.m_extentWidthBytes == 0)
                    {
                        return;
                    }

                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);

                    if(iDstDev == iSrcDev)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    {
                        auto const & uniformedCudaHipMemCpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyAsync(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                uniformedCudaHipMemCpyKind,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);

                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpyPeerAsync(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes),
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    ALPAKA_CUDA_RT_CHECK(
                        cudaStreamSynchronize(
                            queue.m_spQueueImpl->m_UniformedCudaHipQueue));
#else
                    {
                        auto const & uniformedCudaHipMemCpyKind(task.m_hipMemCpyKind);

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpyAsync(
                                dstNativePtr,
                                srcNativePtr,
                                static_cast<std::size_t>(extentWidthBytes),
                                uniformedCudaHipMemCpyKind,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        // Initiate the memory copy.
                        ALPAKA_HIP_RT_CHECK(
                            hipMemcpyPeerAsync(
                                dstNativePtr,
                                iDstDev,
                                srcNativePtr,
                                iSrcDev,
                                static_cast<std::size_t>(extentWidthBytes),
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }

                    ALPAKA_HIP_RT_CHECK( hipStreamSynchronize(
                        queue.m_spQueueImpl->m_UniformedCudaHipQueue));
#endif
                }
            };
            //#############################################################################
            //! The CUDA-HIP non-blocking device queue 2D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformedCudaHipRtNonBlocking,
                mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformedCudaHipRtNonBlocking & queue,
                    mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    // This is not only an optimization but also prevents a division by zero.
                    if(task.m_extentWidthBytes == 0 || task.m_extentHeight == 0)
                    {
                        return;
                    }

                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        auto const & cudaMemcpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2DAsync(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                srcNativePtr,
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                cudaMemcpyKind,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);

                        // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::uniformedCudaHip::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
#else
                    auto const & hipMemCpyKind(task.m_hipMemCpyKind);

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            iDstDev));

                    if(iDstDev != iSrcDev)
                    {
                        // HIP relies on unified memory, so memcpy commands automatically do device-to-device transfers.
                        // P2P access has to be enabled to avoid host transfer.
                        // Checks if devices are connected via PCIe switch and enable P2P access then.
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);
                    }

                    ALPAKA_HIP_RT_CHECK(
                        hipMemcpy2DAsync(
                            dstNativePtr,
                            static_cast<std::size_t>(dstPitchBytesX),
                            srcNativePtr,
                            static_cast<std::size_t>(srcPitchBytesX),
                            static_cast<std::size_t>(extentWidthBytes),
                            static_cast<std::size_t>(extentHeight),
                            hipMemCpyKind,
                            queue.m_spQueueImpl->m_UniformedCudaHipQueue));

                    ALPAKA_HIP_RT_CHECK( hipStreamSynchronize(
                        queue.m_spQueueImpl->m_UniformedCudaHipQueue));
#endif
                }
            };
            //#############################################################################
            //! The CUDA-HIP blocking device queue 2D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformedCudaHipRtBlocking,
                mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformedCudaHipRtBlocking & queue,
                    mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<2u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    // This is not only an optimization but also prevents a division by zero.
                    if(task.m_extentWidthBytes == 0 || task.m_extentHeight == 0)
                    {
                        return;
                    }

                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    if(iDstDev == iSrcDev)
                    {
                        auto const & extentWidthBytes(task.m_extentWidthBytes);
                        auto const & extentHeight(task.m_extentHeight);

                        auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                        auto const & srcPitchBytesX(task.m_srcpitchBytesX);

                        auto const & dstNativePtr(task.m_dstMemNative);
                        auto const & srcNativePtr(task.m_srcMemNative);
                        auto const & cudaMemcpyKind(task.m_cudaMemCpyKind);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy2DAsync(
                                dstNativePtr,
                                static_cast<std::size_t>(dstPitchBytesX),
                                srcNativePtr,
                                static_cast<std::size_t>(srcPitchBytesX),
                                static_cast<std::size_t>(extentWidthBytes),
                                static_cast<std::size_t>(extentHeight),
                                cudaMemcpyKind,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);

                        // There is no cudaMemcpy2DPeerAsync, therefore we use cudaMemcpy3DPeerAsync.
                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::uniformedCudaHip::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    ALPAKA_CUDA_RT_CHECK(
                        cudaStreamSynchronize(
                            queue.m_spQueueImpl->m_UniformedCudaHipQueue));
#else
                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

                    auto const & extentWidthBytes(task.m_extentWidthBytes);
                    auto const & extentHeight(task.m_extentHeight);

                    auto const & dstPitchBytesX(task.m_dstpitchBytesX);
                    auto const & srcPitchBytesX(task.m_srcpitchBytesX);

                    auto const & dstNativePtr(task.m_dstMemNative);
                    auto const & srcNativePtr(task.m_srcMemNative);
                    auto const & hipMemCpyKind(task.m_hipMemCpyKind);

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            iDstDev));

                    if(iDstDev != iSrcDev)
                    {
                        // HIP relies on unified memory, so memcpy commands automatically do device-to-device transfers.
                        // P2P access has to be enabled to avoid host transfer.
                        // Checks if devices are connected via PCIe switch and enable P2P access then.
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);
                    }

                    ALPAKA_HIP_RT_CHECK(
                        hipMemcpy2DAsync(
                            dstNativePtr,
                            static_cast<std::size_t>(dstPitchBytesX),
                            srcNativePtr,
                            static_cast<std::size_t>(srcPitchBytesX),
                            static_cast<std::size_t>(extentWidthBytes),
                            static_cast<std::size_t>(extentHeight),
                            hipMemCpyKind,
                            queue.m_spQueueImpl->m_UniformedCudaHipQueue));

                    ALPAKA_HIP_RT_CHECK( hipStreamSynchronize(
                        queue.m_spQueueImpl->m_UniformedCudaHipQueue));
#endif
                }
            };
            //#############################################################################
            //! The CUDA-HIP non-blocking device queue 3D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformedCudaHipRtNonBlocking,
                mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformedCudaHipRtNonBlocking & queue,
                    mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    // This is not only an optimization but also prevents a division by zero.
                    if(task.m_extentWidthBytes == 0 || task.m_extentHeight == 0 || task.m_extentDepth == 0)
                    {
                        return;
                    }

                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                    // Create the struct describing the copy.
                    hipMemcpy3DParms const hipMemCpy3DParms(
                        mem::view::uniformedCudaHip::detail::buildUniformedCudaHipMemcpy3DParms(
                            task));
                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            iDstDev));
#endif
                    if(iDstDev == iSrcDev)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DParms const uniformedCudaHipMemCpy3DParms(
                            mem::view::uniformedCudaHip::detail::buildUniformedCudaHipMemcpy3DParms(
                                task));
                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DAsync(
                                &uniformedCudaHipMemCpy3DParms,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);

                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::uniformedCudaHip::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
#else
                    {
                        // HIP relies on unified memory, so memcpy commands automatically do device-to-device transfers.
                        // P2P access has to be enabled to avoid host transfer.
                        // Checks if devices are connected via PCIe switch and enable P2P access then.
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);
                    }

                    // Initiate the memory copy.
                    ALPAKA_HIP_RT_CHECK(
                        hipMemcpy3DAsync(
                            &hipMemCpy3DParms,
                            queue.m_spQueueImpl->m_UniformedCudaHipQueue));                }
#endif
                }
            };
            //#############################################################################
            //! The CUDA-HIP blocking device queue 3D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueUniformedCudaHipRtBlocking,
                mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueUniformedCudaHipRtBlocking & queue,
                    mem::view::uniformedCudaHip::detail::TaskCopyUniformedCudaHip<dim::DimInt<3u>, TViewDst, TViewSrc, TExtent> const & task)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    task.printDebug();
#endif
                    // This is not only an optimization but also prevents a division by zero.
                    if(task.m_extentWidthBytes == 0 || task.m_extentHeight == 0 || task.m_extentDepth == 0)
                    {
                        return;
                    }

                    auto const & iDstDev(task.m_iDstDevice);
                    auto const & iSrcDev(task.m_iSrcDevice);
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                    hipMemcpy3DParms const hipMemCpy3DParms(
                        mem::view::uniformedCudaHip::detail::buildUniformedCudaHipMemcpy3DParms(
                            task));
                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            iDstDev));
#endif
                    if(iDstDev == iSrcDev)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    {
                        // Create the struct describing the copy.
                        cudaMemcpy3DParms const uniformedCudaHipMemCpy3DParms(
                            mem::view::uniformedCudaHip::detail::buildUniformedCudaHipMemcpy3DParms(
                                task));
                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                iDstDev));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DAsync(
                                &uniformedCudaHipMemCpy3DParms,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    else
                    {
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);

                        // Create the struct describing the copy.
                        cudaMemcpy3DPeerParms const cudaMemCpy3DPeerParms(
                            mem::view::uniformedCudaHip::detail::buildCudaMemcpy3DPeerParms(
                                task));
                        // Initiate the memory copy.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMemcpy3DPeerAsync(
                                &cudaMemCpy3DPeerParms,
                                queue.m_spQueueImpl->m_UniformedCudaHipQueue));
                    }
                    ALPAKA_CUDA_RT_CHECK(
                        cudaStreamSynchronize(
                            queue.m_spQueueImpl->m_UniformedCudaHipQueue));
#else
                    {
                        // HIP relies on unified memory, so memcpy commands automatically do device-to-device transfers.
                        // P2P access has to be enabled to avoid host transfer.
                        // Checks if devices are connected via PCIe switch and enable P2P access then.
                        alpaka::mem::view::uniformedCudaHip::detail::enablePeerAccessIfPossible(iSrcDev, iDstDev);
                    }

                    // Initiate the memory copy.
                    ALPAKA_HIP_RT_CHECK(
                        hipMemcpy3DAsync(
                            &hipMemCpy3DParms,
                            queue.m_spQueueImpl->m_UniformedCudaHipQueue));

                    ALPAKA_HIP_RT_CHECK( hipStreamSynchronize(
                        queue.m_spQueueImpl->m_UniformedCudaHipQueue));
#endif
                }
            };
        }
    }
}

#endif
