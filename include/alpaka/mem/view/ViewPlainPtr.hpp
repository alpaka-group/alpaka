/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/mem/view/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevCudaRt.hpp>

namespace alpaka
{
    namespace mem
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
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent>
                ALPAKA_FN_HOST_ACC ViewPlainPtr(
                    TElem * pMem,
                    TDev const & dev,
                    TExtent const & extent = TExtent()) :
                        m_pMem(pMem),
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_pitchBytes(calculatePitchesFromExtents(m_extentElements))
                {}

                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent,
                    typename TPitch>
                ALPAKA_FN_HOST_ACC ViewPlainPtr(
                    TElem * pMem,
                    TDev const dev,
                    TExtent const & extent,
                    TPitch const & pitchBytes) :
                        m_pMem(pMem),
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_pitchBytes(
                            vec::subVecEnd<TDim>(
                               static_cast<
                                    vec::Vec<TDim, TIdx> >(pitchBytes)
                            )
                        )
                {}

                //-----------------------------------------------------------------------------
                ViewPlainPtr(ViewPlainPtr const &) = delete;
                //-----------------------------------------------------------------------------
                ViewPlainPtr(ViewPlainPtr &&) = default;
                //-----------------------------------------------------------------------------
                auto operator=(ViewPlainPtr const &) -> ViewPlainPtr & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(ViewPlainPtr &&) -> ViewPlainPtr & = default;
                //-----------------------------------------------------------------------------
                ~ViewPlainPtr() = default;

            private:
                //-----------------------------------------------------------------------------
                //! Calculate the pitches purely from the extents.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent>
                ALPAKA_FN_HOST_ACC static auto calculatePitchesFromExtents(
                    TExtent const & extent)
                -> vec::Vec<TDim, TIdx>
                {
                    vec::Vec<TDim, TIdx> pitchBytes(vec::Vec<TDim, TIdx>::all(0));
                    pitchBytes[TDim::value - 1u] = extent[TDim::value - 1u] * static_cast<TIdx>(sizeof(TElem));
                    for(TIdx i = TDim::value - 1u; i > static_cast<TIdx>(0u); --i)
                    {
                        pitchBytes[i-1] = extent[i-1] * pitchBytes[i];
                    }
                    return pitchBytes;
                }

            public:
                TElem * const m_pMem;
                TDev const m_dev;
                vec::Vec<TDim, TIdx> const m_extentElements;
                vec::Vec<TDim, TIdx> const m_pitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for ViewPlainPtr.
    namespace dev
    {
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
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The ViewPlainPtr device get trait specialization.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetDev<
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getDev(
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & view)
                    -> TDev
                {
                    return view.m_dev;
                }
            };
        }
    }
    namespace dim
    {
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
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
            {
                using type = TDim;
            };
        }
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
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
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
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>,
                typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & extent)
                -> TIdx
                {
                    return extent.m_extentElements[TIdxIntegralConst::value];
                }
            };
        }
    }
    namespace mem
    {
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
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & view)
                    -> TElem const *
                    {
                        return view.m_pMem;
                    }
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> & view)
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
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>,
                    typename std::enable_if<TIdxIntegralConst::value < TDim::value>::type>
                {
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & view)
                    -> TIdx
                    {
                        return view.m_pitchBytes[TIdxIntegralConst::value];
                    }
                };

                //#############################################################################
                //! The CPU device CreateStaticDevMemView trait specialization.
                template<>
                struct CreateStaticDevMemView<
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem,
                        typename TExtent>
                    ALPAKA_FN_HOST static auto createStaticDevMemView(
                        TElem * pMem,
                        dev::DevCpu const & dev,
                        TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> alpaka::mem::view::ViewPlainPtr<dev::DevCpu, TElem, alpaka::dim::Dim<TExtent>, alpaka::idx::Idx<TExtent>>
#endif
                    {
                        return
                            alpaka::mem::view::ViewPlainPtr<
                                dev::DevCpu,
                                TElem,
                                alpaka::dim::Dim<TExtent>,
                                alpaka::idx::Idx<TExtent>>(
                                    pMem,
                                    dev,
                                    extent);
                    }
                };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                //#############################################################################
                //! The CUDA RT device CreateStaticDevMemView trait specialization.
                template<>
                struct CreateStaticDevMemView<
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem,
                        typename TExtent>
                    ALPAKA_FN_HOST static auto createStaticDevMemView(
                        TElem * pMem,
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> alpaka::mem::view::ViewPlainPtr<dev::DevCudaRt, TElem, alpaka::dim::Dim<TExtent>, alpaka::idx::Idx<TExtent>>
#endif
                    {
                        TElem* pMemAcc(nullptr);
                        ALPAKA_CUDA_RT_CHECK(
                            cudaGetSymbolAddress(
                                reinterpret_cast<void **>(&pMemAcc),
                                *pMem));
                        return
                            alpaka::mem::view::ViewPlainPtr<
                                dev::DevCudaRt,
                                TElem,
                                alpaka::dim::Dim<TExtent>,
                                alpaka::idx::Idx<TExtent>>(
                                    pMemAcc,
                                    dev,
                                    extent);
                    }
                };
#endif
            }
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
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const &)
                -> TIdx
                {
                    return 0u;
                }
            };
        }
    }
    namespace idx
    {
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
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}
