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

#include <alpaka/devs/cpu/mem/Set.hpp>  // Set
#include <alpaka/devs/cpu/Dev.hpp>      // DevCpu

#include <alpaka/traits/mem/Buf.hpp>    // traits::Alloc, ...
#include <alpaka/traits/Extent.hpp>     // traits::getXXX

#include <alpaka/core/mem/View.hpp>     // View
#include <alpaka/core/BasicDims.hpp>    // dim::Dim<N>
#include <alpaka/core/Vec.hpp>          // Vec<TDim>

// \TODO: Remove CUDA inclusion for BufCpu by replacing pinning with non CUDA code!
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #include <alpaka/core/Cuda.hpp>
#endif

#include <cassert>                      // assert
#include <memory>                       // std::shared_ptr

namespace alpaka
{
    namespace devs
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU memory buffer.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                class BufCpu
                {
                private:
                    using Elem = TElem;
                    using Dim = TDim;

                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents>
                    ALPAKA_FCT_HOST BufCpu(
                        devs::cpu::DevCpu const & dev,
                        TExtents const & extents) :
                            m_Dev(dev),
                            m_vExtentsElements(extent::getExtentsVecNd<TDim, UInt>(extents)),
                            m_spMem(
                                reinterpret_cast<TElem *>(
                                    boost::alignment::aligned_alloc(16u, sizeof(TElem) * computeElementCount(extents))), &BufCpu::freeBuffer),
                            m_uiPitchBytes(extent::getWidth<UInt>(extents) * sizeof(TElem))
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " e: " << m_vExtentsElements
                            << " ptr: " << static_cast<void *>(m_spMem.get())
                            << " pitch: " << m_uiPitchBytes
                            << std::endl;
    #endif
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! \return The number of elements to allocate.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents>
                    ALPAKA_FCT_HOST static auto computeElementCount(
                        TExtents const & extents)
                    -> UInt
                    {
                        auto const uiExtentsElementCount(extent::getProductOfExtents<UInt>(extents));
                        assert(uiExtentsElementCount>0);

                        return uiExtentsElementCount;
                    }
                    //-----------------------------------------------------------------------------
                    //! Frees the shared buffer.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto freeBuffer(
                        TElem * pBuffer)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        assert(pBuffer);
                        boost::alignment::aligned_free(
                            reinterpret_cast<void *>(pBuffer));
                    }

                public:
                    devs::cpu::DevCpu m_Dev;
                    Vec<TDim> m_vExtentsElements;
                    std::shared_ptr<TElem> m_spMem;
                    UInt m_uiPitchBytes;
                };
            }
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dev
        {
            //#############################################################################
            //! The BufCuda device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DevType<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                using type = devs::cpu::DevCpu;
            };
            //#############################################################################
            //! The BufCpu device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetDev<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf)
                -> devs::cpu::DevCpu
                {
                    return buf.m_Dev;
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The BufCpu dimension getter trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DimType<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The BufCpu width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim>
            struct GetExtent<
                TIdx,
                devs::cpu::detail::BufCpu<TElem, TDim>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & extents)
                -> UInt
                {
                    return extents.m_vExtentsElements[TIdx::value];
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The CPU device memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct BufType<
                TElem,
                TDim,
                devs::cpu::DevCpu>
            {
                using type = devs::cpu::detail::BufCpu<TElem, TDim>;
            };
            //#############################################################################
            //! The CPU device memory view type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct ViewType<
                TElem,
                TDim,
                devs::cpu::DevCpu>
            {
                using type = alpaka::mem::detail::View<TElem, TDim, devs::cpu::DevCpu>;
            };
            //#############################################################################
            //! The BufCpu memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct ElemType<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                using type = TElem;
            };
            //#############################################################################
            //! The BufCpu buf trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetBuf<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf)
                -> devs::cpu::detail::BufCpu<TElem, TDim> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBuf(
                    devs::cpu::detail::BufCpu<TElem, TDim> & buf)
                -> devs::cpu::detail::BufCpu<TElem, TDim> &
                {
                    return buf;
                }
            };
            //#############################################################################
            //! The BufCpu native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPtrNative<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf)
                -> TElem const *
                {
                    return buf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    devs::cpu::detail::BufCpu<TElem, TDim> & buf)
                -> TElem *
                {
                    return buf.m_spMem.get();
                }
            };
            //#############################################################################
            //! The BufCpu pointer on device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPtrDev<
                devs::cpu::detail::BufCpu<TElem, TDim>,
                devs::cpu::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrDev(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf,
                    devs::cpu::DevCpu const & dev)
                -> TElem const *
                {
                    if(dev == alpaka::dev::getDev(buf))
                    {
                        return buf.m_spMem.get();
                    }
                    else
                    {
                        throw std::runtime_error("The buffer is not accessible from the given device!");
                    }
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrDev(
                    devs::cpu::detail::BufCpu<TElem, TDim> & buf,
                    devs::cpu::DevCpu const & dev)
                -> TElem *
                {
                    if(dev == alpaka::dev::getDev(buf))
                    {
                        return buf.m_spMem.get();
                    }
                    else
                    {
                        throw std::runtime_error("The buffer is not accessible from the given device!");
                    }
                }
            };
            //#############################################################################
            //! The BufCpu pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                std::integral_constant<UInt, TDim::value - 1u>,
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPitchBytes(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & pitch)
                -> UInt
                {
                    return pitch.m_uiPitchBytes;
                }
            };
            //#############################################################################
            //! The CPU device memory allocation trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Alloc<
                TElem,
                TDim,
                devs::cpu::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    devs::cpu::DevCpu const & dev,
                    TExtents const & extents)
                -> devs::cpu::detail::BufCpu<TElem, TDim>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    return devs::cpu::detail::BufCpu<
                        TElem,
                        TDim>(
                            dev,
                            extents);
                }
            };
            //#############################################################################
            //! The CPU device memory mapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Map<
                devs::cpu::detail::BufCpu<TElem, TDim>,
                devs::cpu::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto map(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf,
                    devs::cpu::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        throw std::runtime_error("Memory mapping of BufCpu between two devices is not implemented!");
                    }
                    // If it is the same device, nothing has to be mapped.
                }
            };
            //#############################################################################
            //! The CPU device memory unmapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Unmap<
                devs::cpu::detail::BufCpu<TElem, TDim>,
                devs::cpu::DevCpu>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto unmap(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf,
                    devs::cpu::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        throw std::runtime_error("Memory unmapping of BufCpu between two devices is not implemented!");
                    }
                    // If it is the same device, nothing has to be mapped.
                }
            };
            //#############################################################################
            //! The CPU device memory pinning trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Pin<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto pin(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
                    // - cudaHostRegisterDefault:
                    //   See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                    // - cudaHostRegisterPortable:
                    //   The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        cudaHostRegister(
                            const_cast<void *>(reinterpret_cast<void const *>(alpaka::mem::getPtrNative(buf))),
                            alpaka::extent::getProductOfExtents<std::size_t>(buf) * sizeof(alpaka::mem::ElemT<devs::cpu::detail::BufCpu<TElem, TDim>>),
                            cudaHostRegisterDefault),
                        cudaErrorHostMemoryAlreadyRegistered);
#else
                    throw std::runtime_error("Memory pinning of BufCpu is not implemented when CUDA is not enabled!");
#endif
                }
            };
            //#############################################################################
            //! The CPU device memory unpinning trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Unpin<
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto unpin(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
                    ALPAKA_CUDA_RT_CHECK_IGNORE(
                        cudaHostUnregister(
                            const_cast<void *>(reinterpret_cast<void const *>(alpaka::mem::getPtrNative(buf)))),
                        cudaErrorHostMemoryNotRegistered);
#else
                    throw std::runtime_error("Memory unpinning of BufCpu is not implemented when CUDA is not enabled!");
#endif
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The BufCpu offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim>
            struct GetOffset<
                TIdx,
                devs::cpu::detail::BufCpu<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    devs::cpu::detail::BufCpu<TElem, TDim> const &)
                -> UInt
                {
                    return 0u;
                }
            };
        }
    }
}
