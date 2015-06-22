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

#include <alpaka/dev/DevCudaRt.hpp>         // DevCudaRt
#include <alpaka/dim/DimIntegralConst.hpp>  // dim::Dim<N>
#include <alpaka/extent/Traits.hpp>         // view::getXXX
#include <alpaka/mem/buf/BufCpu.hpp>        // BufCpu
#include <alpaka/mem/buf/Traits.hpp>        // view::Copy, ...
#include <alpaka/mem/view/ViewBasic.hpp>    // ViewBasic

#include <alpaka/core/Vec.hpp>              // Vec<TDim>
#include <alpaka/core/Cuda.hpp>             // cudaMalloc, ...

#include <cassert>                          // assert
#include <memory>                           // std::shared_ptr

namespace alpaka
{
    namespace mem
    {
        namespace buf
        {
            //#############################################################################
            //! The CUDA memory buffer.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            class BufCudaRt
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
                ALPAKA_FCT_HOST BufCudaRt(
                    dev::DevCudaRt dev,
                    TElem * const pMem,
                    UInt const & uiPitchBytes,
                    TExtents const & extents) :
                        m_Dev(dev),
                        m_vExtentsElements(extent::getExtentsVecNd<TDim, UInt>(extents)),
                        m_spMem(
                            pMem,
                            std::bind(&BufCudaRt::freeBuffer, std::placeholders::_1, std::ref(m_Dev))),
                        m_uiPitchBytes(uiPitchBytes)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        TDim::value == dim::DimT<TExtents>::value,
                        "The extents are required to have the same dimensionality as the BufCudaRt!");
                }

            private:
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto freeBuffer(
                    TElem * pBuffer,
                    dev::DevCudaRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    assert(pBuffer);

                    // Set the current device. \TODO: Is setting the current device before cudaFree required?
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    // Free the buffer.
                    cudaFree(reinterpret_cast<void *>(pBuffer));
                }

            public:
                dev::DevCudaRt m_Dev;
                Vec<TDim> m_vExtentsElements;
                std::shared_ptr<TElem> m_spMem;
                UInt m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCudaRt.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCudaRt device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DevType<
                mem::buf::BufCudaRt<TElem, TDim>>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The BufCudaRt device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetDev<
                mem::buf::BufCudaRt<TElem, TDim>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    mem::buf::BufCudaRt<TElem, TDim> const & buf)
                -> dev::DevCudaRt
                {
                    return buf.m_Dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCudaRt dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DimType<
                mem::buf::BufCudaRt<TElem, TDim>>
            {
                using type = TDim;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCudaRt extent get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim>
            struct GetExtent<
                TIdx,
                mem::buf::BufCudaRt<TElem, TDim>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    mem::buf::BufCudaRt<TElem, TDim> const & extent)
                -> UInt
                {
                    return extent.m_vExtentsElements[TIdx::value];
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
                //! The BufCudaRt memory buffer type trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct ViewType<
                    TElem,
                    TDim,
                    dev::DevCudaRt>
                {
                    using type = mem::view::ViewBasic<TElem, TDim, dev::DevCudaRt>;
                };
                //#############################################################################
                //! The BufCudaRt memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct ElemType<
                    mem::buf::BufCudaRt<TElem, TDim>>
                {
                    using type = TElem;
                };
                //#############################################################################
                //! The BufCudaRt buf trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct GetBuf<
                    mem::buf::BufCudaRt<TElem, TDim>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getBuf(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf)
                    -> mem::buf::BufCudaRt<TElem, TDim> const &
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getBuf(
                        mem::buf::BufCudaRt<TElem, TDim> & buf)
                    -> mem::buf::BufCudaRt<TElem, TDim> &
                    {
                        return buf;
                    }
                };
                //#############################################################################
                //! The BufCudaRt native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct GetPtrNative<
                    mem::buf::BufCudaRt<TElem, TDim>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrNative(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf)
                    -> TElem const *
                    {
                        return buf.m_spMem.get();
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrNative(
                        mem::buf::BufCudaRt<TElem, TDim> & buf)
                    -> TElem *
                    {
                        return buf.m_spMem.get();
                    }
                };
                //#############################################################################
                //! The BufCudaRt pointer on device get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct GetPtrDev<
                    mem::buf::BufCudaRt<TElem, TDim>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrDev(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf,
                        dev::DevCudaRt const & dev)
                    -> TElem const *
                    {
                        if(dev == dev::getDev(buf))
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
                        mem::buf::BufCudaRt<TElem, TDim> & buf,
                        dev::DevCudaRt const & dev)
                    -> TElem *
                    {
                        if(dev == dev::getDev(buf))
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
                //! The BufCudaRt pitch get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct GetPitchBytes<
                    std::integral_constant<UInt, TDim::value - 1u>,
                    mem::buf::BufCudaRt<TElem, TDim>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPitchBytes(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf)
                    -> UInt
                    {
                        return buf.m_uiPitchBytes;
                    }
                };
            }
        }
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCudaRt memory buffer type trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct BufType<
                    TElem,
                    TDim,
                    dev::DevCudaRt>
                {
                    using type = mem::buf::BufCudaRt<TElem, TDim>;
                };
                //#############################################################################
                //! The CUDA 1D memory allocation trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct Alloc<
                    T,
                    dim::Dim1,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents>
                    ALPAKA_FCT_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtents const & extents)
                    -> mem::buf::BufCudaRt<T, dim::Dim1>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const uiWidth(extent::getWidth<UInt>(extents));
                        assert(uiWidth>0);
                        auto const uiWidthBytes(uiWidth * sizeof(T));
                        assert(uiWidthBytes>0);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev.m_iDevice));
                        // Allocate the buffer on this device.
                        void * pBuffer;
                        ALPAKA_CUDA_RT_CHECK(cudaMalloc(
                            &pBuffer,
                            uiWidthBytes));
                        assert((pBuffer));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiWidth
                            << " ewb: " << uiWidthBytes
                            << " ptr: " << pBuffer
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<T, dim::Dim1>(
                                dev,
                                reinterpret_cast<T *>(pBuffer),
                                uiWidthBytes,
                                extents);
                    }
                };
                //#############################################################################
                //! The CUDA 2D memory allocation trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct Alloc<
                    T,
                    dim::Dim2,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents>
                    ALPAKA_FCT_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtents const & extents)
                    -> mem::buf::BufCudaRt<T, dim::Dim2>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const uiWidth(extent::getWidth<UInt>(extents));
                        auto const uiWidthBytes(uiWidth * sizeof(T));
                        assert(uiWidthBytes>0);
                        auto const uiHeight(extent::getHeight<UInt>(extents));
#ifndef NDEBUG
                        auto const uiElementCount(uiWidth * uiHeight);
#endif
                        assert(uiElementCount>0);

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev.m_iDevice));
                        // Allocate the buffer on this device.
                        void * pBuffer;
                        std::size_t uiPitch;
                        ALPAKA_CUDA_RT_CHECK(cudaMallocPitch(
                            &pBuffer,
                            &uiPitch,
                            uiWidthBytes,
                            uiHeight));
                        assert(pBuffer);
                        assert(uiPitch>=uiWidthBytes);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << uiWidth
                            << " eh: " << uiHeight
                            << " ewb: " << uiWidthBytes
                            << " ptr: " << pBuffer
                            << " pitch: " << uiPitch
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<T, dim::Dim2>(
                                dev,
                                reinterpret_cast<T *>(pBuffer),
                                static_cast<UInt>(uiPitch),
                                extents);
                    }
                };
                //#############################################################################
                //! The CUDA 3D memory allocation trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct Alloc<
                    T,
                    dim::Dim3,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtents>
                    ALPAKA_FCT_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtents const & extents)
                    -> mem::buf::BufCudaRt<T, dim::Dim3>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        cudaExtent const cudaExtentVal(
                            make_cudaExtent(
                                extent::getWidth<UInt>(extents) * sizeof(T),
                                extent::getHeight<UInt>(extents),
                                extent::getDepth<UInt>(extents)));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                            dev.m_iDevice));
                        // Allocate the buffer on this device.
                        cudaPitchedPtr cudaPitchedPtrVal;
                        ALPAKA_CUDA_RT_CHECK(cudaMalloc3D(
                            &cudaPitchedPtrVal,
                            cudaExtentVal));

                        assert(cudaPitchedPtrVal.ptr);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << extent::getWidth<UInt>(extents)
                            << " eh: " << cudaExtentVal.height
                            << " ed: " << cudaExtentVal.depth
                            << " ewb: " << cudaExtentVal.width
                            << " ptr: " << cudaPitchedPtrVal.ptr
                            << " pitch: " << cudaPitchedPtrVal.pitch
                            << " wb: " << cudaPitchedPtrVal.xsize
                            << " h: " << cudaPitchedPtrVal.ysize
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<T, dim::Dim3>(
                                dev,
                                reinterpret_cast<T *>(cudaPitchedPtrVal.ptr),
                                static_cast<UInt>(cudaPitchedPtrVal.pitch),
                                extents);
                    }
                };
                //#############################################################################
                //! The BufCudaRt CUDA device memory mapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct Map<
                    mem::buf::BufCudaRt<TElem, TDim>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto map(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Mapping memory from one CUDA device into an other CUDA device not implemented!");
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCudaRt CUDA device memory unmapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct Unmap<
                    mem::buf::BufCudaRt<TElem, TDim>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto unmap(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Unmapping memory mapped from one CUDA device into an other CUDA device not implemented!");
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
                //#############################################################################
                //! The BufCudaRt memory pinning trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct Pin<
                    mem::buf::BufCudaRt<TElem, TDim>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto pin(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // CUDA device memory is always pinned, it can not be swapped out.
                    }
                };
                //#############################################################################
                //! The BufCudaRt memory unpinning trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct Unpin<
                    mem::buf::BufCudaRt<TElem, TDim>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto unpin(
                        mem::buf::BufCudaRt<TElem, TDim> const & buf)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // CUDA device memory is always pinned, it can not be swapped out.
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCudaRt offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TDim>
            struct GetOffset<
                TIdx,
                mem::buf::BufCudaRt<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                   mem::buf::BufCudaRt<TElem, TDim> const &)
                -> UInt
                {
                    return 0u;
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    //-----------------------------------------------------------------------------
    namespace mem
    {
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu CUDA device memory mapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct Map<
                    mem::buf::BufCpu<TElem, TDim>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim> const & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // cudaHostRegisterMapped:
                            //   Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                            //   This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                            ALPAKA_CUDA_RT_CHECK(
                                cudaHostRegister(
                                    const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                    extent::getProductOfExtents<std::size_t>(buf) * sizeof(mem::view::ElemT<BufCpu<TElem, TDim>>),
                                    cudaHostRegisterMapped));
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCpu CUDA device memory unmapping trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct Unmap<
                    mem::buf::BufCpu<TElem, TDim>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim> const & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // Unmaps the memory range whose base address is specified by ptr, and makes it pageable again.
                            // \FIXME: If the memory has separately been pinned before we destroy the pinning state.
                            ALPAKA_CUDA_RT_CHECK(
                                cudaHostUnregister(
                                    const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf)))));
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
            }
        }
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu pointer on CUDA device get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                struct GetPtrDev<
                    mem::buf::BufCpu<TElem, TDim>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim> const & buf,
                        dev::DevCudaRt const & dev)
                    -> TElem const *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_CUDA_RT_CHECK(
                            cudaHostGetDevicePointer(
                                &pDev,
                                const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                0));
                        return pDev;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim> & buf,
                        dev::DevCudaRt const & dev)
                    -> TElem *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_CUDA_RT_CHECK(
                            cudaHostGetDevicePointer(
                                &pDev,
                                mem::view::getPtrNative(buf),
                                0));
                        return pDev;
                    }
                };
            }
        }
    }
}

#include <alpaka/mem/buf/cuda/Copy.hpp>
#include <alpaka/mem/buf/cuda/Set.hpp>
