/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/queue/QueueOmp4Blocking.hpp>

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevOmp4.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Omp4.hpp>

#include <set>
#include <tuple>

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace omp4
            {
                namespace detail
                {
                    //#############################################################################
                    //! The Omp4 memory copy trait.
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyOmp4;

                    //#############################################################################
                    //! The 1D Omp4 memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyOmp4<
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
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyOmp4(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
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
                }
            }

            //-----------------------------------------------------------------------------
            // Trait specializations for CreateTaskCopy.
            namespace traits
            {
                namespace omp4
                {
                    namespace detail
                    {
                        //#############################################################################
                        //! The Omp4 memory copy task creation trait detail.
                        template<
                            typename TDim,
                            typename TDevDst,
                            typename TDevSrc>
                        struct CreateTaskCopyImpl
                        {
                            //-----------------------------------------------------------------------------
                            template<
                                typename TExtent,
                                typename TViewSrc,
                                typename TViewDst>
                            ALPAKA_FN_HOST static auto createTaskCopy(
                                TViewDst & viewDst,
                                TViewSrc const & viewSrc,
                                TExtent const & extent,
                                int iDeviceDst = 0,
                                int iDeviceSrc = 0
                                )
                            -> mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>
                            {
                                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                                return
                                    mem::view::omp4::detail::TaskCopyOmp4<
                                        TDim,
                                        TViewDst,
                                        TViewSrc,
                                        TExtent>(
                                            viewDst,
                                            viewSrc,
                                            extent,
                                            iDeviceDst,
                                            iDeviceSrc);
                            }
                        };
                    }
                }

                //#############################################################################
                //! The CPU to Omp4 memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevOmp4,
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
                    -> mem::view::omp4::detail::TaskCopyOmp4<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    dev::getDev(viewDst).m_iDevice,
                                    0
                                    );
                    }
                };

                //#############################################################################
                //! The Omp4 to CPU memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevOmp4>
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
                    -> mem::view::omp4::detail::TaskCopyOmp4<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    0,
                                    dev::getDev(viewSrc).m_iDevice
                                    );
                    }
                };

                //#############################################################################
                //! The Omp4 to Omp4 memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevOmp4,
                    dev::DevOmp4>
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
                    -> mem::view::omp4::detail::TaskCopyOmp4<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    dev::getDev(viewDst).m_iDevice,
                                    dev::getDev(viewSrc).m_iDevice
                                    );
                    }
                };
            }
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The Omp4 blocking device queue 1D copy enqueue trait specialization.
            template<
                typename TExtent,
                typename TViewSrc,
                typename TViewDst>
            struct Enqueue<
                queue::QueueOmp4Blocking,
                mem::view::omp4::detail::TaskCopyOmp4<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueOmp4Blocking & queue,
                    mem::view::omp4::detail::TaskCopyOmp4<dim::DimInt<1u>, TViewDst, TViewSrc, TExtent> const & task)
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

                    alpaka::ignore_unused(queue); //! \TODO

                    ALPAKA_OMP4_CHECK(
                        omp_target_memcpy(
                            dstNativePtr, const_cast<void*>(srcNativePtr), extentWidthBytes, 0,0, iDstDev, iSrcDev));
                }
            };
        }
    }
}

#endif
