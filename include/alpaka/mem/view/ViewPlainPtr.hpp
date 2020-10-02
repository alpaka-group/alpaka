/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/mem/view/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevUniformCudaHipRt.hpp>
#include <alpaka/dev/DevOmp5.hpp>

#include <type_traits>

namespace alpaka
{
    namespace view
    {
        //#############################################################################
        //! The memory view to wrap plain pointers.
        template<
            typename TDev,
            typename TElem,
            typename TDim,
            typename TIdx>
        class ViewPlainPtr final
        {
            static_assert(
                !std::is_const<TIdx>::value,
                "The idx type of the view can not be const!");

            using Dev = alpaka::Dev<TDev>;
        public:
            //-----------------------------------------------------------------------------
            template<
                typename TExtent>
            ALPAKA_FN_HOST ViewPlainPtr(
                TElem * pMem,
                Dev const & dev,
                TExtent const & extent = TExtent()) :
                    m_pMem(pMem),
                    m_dev(dev),
                    m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                    m_pitchBytes(calculatePitchesFromExtents(m_extentElements))
            {}

            //-----------------------------------------------------------------------------
            template<
                typename TExtent,
                typename TPitch>
            ALPAKA_FN_HOST ViewPlainPtr(
                TElem * pMem,
                Dev const dev,
                TExtent const & extent,
                TPitch const & pitchBytes) :
                    m_pMem(pMem),
                    m_dev(dev),
                    m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                    m_pitchBytes(
                        subVecEnd<TDim>(
                            static_cast<
                                Vec<TDim, TIdx> >(pitchBytes)
                        )
                    )
            {}

            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST
            ViewPlainPtr(ViewPlainPtr const &) = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST
            ViewPlainPtr(ViewPlainPtr && other) noexcept :
                    m_pMem(other.m_pMem),
                    m_dev(other.m_dev),
                    m_extentElements(other.m_extentElements),
                    m_pitchBytes(other.m_pitchBytes)
            {
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST
            auto operator=(ViewPlainPtr const &) -> ViewPlainPtr & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST
            auto operator=(ViewPlainPtr &&) -> ViewPlainPtr & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ViewPlainPtr() = default;

        private:
            //-----------------------------------------------------------------------------
            //! Calculate the pitches purely from the extents.
            template<
                typename TExtent>
            ALPAKA_FN_HOST static auto calculatePitchesFromExtents(
                TExtent const & extent)
            -> Vec<TDim, TIdx>
            {
                Vec<TDim, TIdx> pitchBytes(Vec<TDim, TIdx>::all(0));
                pitchBytes[TDim::value - 1u] = extent[TDim::value - 1u] * static_cast<TIdx>(sizeof(TElem));
                for(TIdx i = TDim::value - 1u; i > static_cast<TIdx>(0u); --i)
                {
                    pitchBytes[i-1] = extent[i-1] * pitchBytes[i];
                }
                return pitchBytes;
            }

        public:
            TElem * const m_pMem;
            Dev const m_dev;
            Vec<TDim, TIdx> const m_extentElements;
            Vec<TDim, TIdx> const m_pitchBytes;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for ViewPlainPtr.
    namespace traits
    {
        //#############################################################################
        //! The ViewPlainPtr device type trait specialization.
        template<
            typename TDev,
            typename TElem,
            typename TDim,
            typename TIdx>
        struct DevType<
            view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            using type = alpaka::Dev<TDev>;
        };

        //#############################################################################
        //! The ViewPlainPtr device get trait specialization.
        template<
            typename TDev,
            typename TElem,
            typename TDim,
            typename TIdx>
        struct GetDev<
            view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            static auto getDev(
                view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & view)
                -> alpaka::Dev<TDev>
            {
                return view.m_dev;
            }
        };
    }
    namespace traits
    {
        //#############################################################################
        //! The ViewPlainPtr dimension getter trait.
        template<
            typename TDev,
            typename TElem,
            typename TDim,
            typename TIdx>
        struct DimType<
            view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            using type = TDim;
        };
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr memory element type get trait specialization.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct ElemType<
                view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
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
            //! The ViewPlainPtr width get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                view::ViewPlainPtr<TDev, TElem, TDim, TIdx>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC
                static auto getExtent(
                    view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & extent)
                -> TIdx
                {
                    return extent.m_extentElements[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace view
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr native pointer get trait specialization.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetPtrNative<
                view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
            {
                static auto getPtrNative(
                    view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & view)
                -> TElem const *
                {
                    return view.m_pMem;
                }
                static auto getPtrNative(
                    view::ViewPlainPtr<TDev, TElem, TDim, TIdx> & view)
                -> TElem *
                {
                    return view.m_pMem;
                }
            };

            //#############################################################################
            //! The ViewPlainPtr memory pitch get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetPitchBytes<
                TIdxIntegralConst,
                view::ViewPlainPtr<TDev, TElem, TDim, TIdx>,
                std::enable_if_t<TIdxIntegralConst::value < TDim::value>>
            {
                ALPAKA_FN_HOST static auto getPitchBytes(
                    view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & view)
                -> TIdx
                {
                    return view.m_pitchBytes[TIdxIntegralConst::value];
                }
            };

            //#############################################################################
            //! The CPU device CreateStaticDevMemView trait specialization.
            template<>
            struct CreateStaticDevMemView<
                DevCpu>
            {
                //-----------------------------------------------------------------------------
                template<
                    typename TElem,
                    typename TExtent>
                static auto createStaticDevMemView(
                    TElem * pMem,
                    DevCpu const & dev,
                    TExtent const & extent)
                {
                    return
                        alpaka::view::ViewPlainPtr<
                            DevCpu,
                            TElem,
                            alpaka::Dim<TExtent>,
                            alpaka::Idx<TExtent>>(
                                pMem,
                                dev,
                                extent);
                }
            };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
            //#############################################################################
            //! The CUDA/HIP RT device CreateStaticDevMemView trait specialization.
            template<>
            struct CreateStaticDevMemView<
                DevUniformCudaHipRt>
            {
                //-----------------------------------------------------------------------------
                template<
                    typename TElem,
                    typename TExtent>
                static auto createStaticDevMemView(
                    TElem * pMem,
                    DevUniformCudaHipRt const & dev,
                    TExtent const & extent)
                {
                    TElem* pMemAcc(nullptr);

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        cudaGetSymbolAddress(
                            reinterpret_cast<void **>(&pMemAcc),
                            *pMem));
#else
#ifdef __HIP_PLATFORM_NVCC__
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipCUDAErrorTohipError(
                        cudaGetSymbolAddress(
                            reinterpret_cast<void **>(&pMemAcc),
                            *pMem)));
#else
                    // FIXME: still does not work in HIP(clang) (results in hipErrorNotFound)
                    // HIP_SYMBOL(X) not useful because it only does #X on HIP(clang), while &X on HIP(NVCC)
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        hipGetSymbolAddress(
                            reinterpret_cast<void **>(&pMemAcc),
                            pMem));
#endif
#endif
                    return
                        alpaka::view::ViewPlainPtr<
                            DevUniformCudaHipRt,
                            TElem,
                            alpaka::Dim<TExtent>,
                            alpaka::Idx<TExtent>>(
                                pMemAcc,
                                dev,
                                extent);
                }
            };
#endif

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
            //#############################################################################
            //! The Omp5 device CreateStaticDevMemView trait specialization.
            //! \todo What ist this for? Does this exist in OMP5?
            template<>
            struct CreateStaticDevMemView<
                DevOmp5>
            {
                //-----------------------------------------------------------------------------
                template<
                    typename TElem,
                    typename TExtent>
                static auto createStaticDevMemView(
                    TElem * pMem,
                    DevOmp5 const & dev,
                    TExtent const & extent)
                {
                    return
                        alpaka::view::ViewPlainPtr<
                            DevOmp5,
                            TElem,
                            alpaka::Dim<TExtent>,
                            alpaka::Idx<TExtent>>(
                                pMem,
                                dev,
                                extent);
                }
            };
#endif
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr offset get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetOffset<
                TIdxIntegralConst,
                view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC
                static auto getOffset(
                    view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const &)
                -> TIdx
                {
                    return 0u;
                }
            };
        }
    }
    namespace traits
    {
        //#############################################################################
        //! The ViewPlainPtr idx type trait specialization.
        template<
            typename TDev,
            typename TElem,
            typename TDim,
            typename TIdx>
        struct IdxType<
            view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };
    }
}
