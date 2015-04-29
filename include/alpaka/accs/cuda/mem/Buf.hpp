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

#include <alpaka/accs/cuda/mem/Set.hpp>     // Set
#include <alpaka/accs/cuda/Dev.hpp>         // DevCuda
#include <alpaka/accs/cuda/Common.hpp>

#include <alpaka/devs/cpu/mem/Buf.hpp>      // BufCpu

#include <alpaka/core/mem/View.hpp>         // View
#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/Vec.hpp>              // Vec<TDim>

#include <alpaka/traits/mem/Buf.hpp>        // traits::Copy, ...
#include <alpaka/traits/Extent.hpp>         // traits::getXXX

#include <cassert>                          // assert
#include <memory>                           // std::shared_ptr

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                //#############################################################################
                //! The CUDA memory buffer.
                //#############################################################################
                template<
                    typename TElem,
                    typename TDim>
                class BufCuda
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
                    ALPAKA_FCT_HOST BufCuda(
                        DevCuda dev,
                        TElem * const pMem,
                        UInt const & uiPitchBytes,
                        TExtents const & extents) :
                            m_Dev(dev),
                            m_vExtentsElements(extent::getExtentsNd<TDim, UInt>(extents)),
                            m_spMem(
                                pMem,
                                std::bind(&BufCuda::freeBuffer, std::placeholders::_1, std::ref(m_Dev))),
                            m_uiPitchBytes(uiPitchBytes)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        static_assert(
                            TDim::value == dim::DimT<TExtents>::value,
                            "The extents are required to have the same dimensionality as the BufCuda!");
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! Frees the shared buffer.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto freeBuffer(
                        TElem * pBuffer,
                        DevCuda const & dev)
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
                    DevCuda m_Dev;
                    Vec<TDim> m_vExtentsElements;
                    std::shared_ptr<TElem> m_spMem;
                    UInt m_uiPitchBytes;
                };
            }
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCuda.
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
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = accs::cuda::detail::DevCuda;
            };

            //#############################################################################
            //! The BufCuda device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetDev<
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf)
                -> accs::cuda::detail::DevCuda
                {
                    return buf.m_Dev;
                }
            };
        }

        namespace dim
        {
            //#############################################################################
            //! The BufCuda dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct DimType<
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The BufCuda extent get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TElem,
                typename TDim>
            struct GetExtent<
                TuiIdx,
                accs::cuda::detail::BufCuda<TElem, TDim>,
                typename std::enable_if<TDim::value >= (TuiIdx+1)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getExtent(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & extent)
                -> UInt
                {
                    return extent.m_vExtentsElements[TuiIdx];
                }
            };
        }

        namespace offset
        {
            //#############################################################################
            //! The BufCuda offset get trait specialization.
            //#############################################################################
            template<
                UInt TuiIdx,
                typename TElem,
                typename TDim>
            struct GetOffset<
                TuiIdx,
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getOffset(
                    accs::cuda::detail::BufCuda<TElem, TDim> const &)
                -> UInt
                {
                    return 0u;
                }
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The BufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct BufType<
                TElem,
                TDim,
                accs::cuda::detail::DevCuda>
            {
                using type = accs::cuda::detail::BufCuda<TElem, TDim>;
            };

            //#############################################################################
            //! The BufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct ViewType<
                TElem,
                TDim,
                accs::cuda::detail::DevCuda>
            {
                using type = alpaka::mem::detail::View<TElem, TDim, accs::cuda::detail::DevCuda>;
            };

            //#############################################################################
            //! The BufCuda memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct ElemType<
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The BufCuda base trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetBase<
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBase(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf)
                -> accs::cuda::detail::BufCuda<TElem, TDim> const &
                {
                    return buf;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getBase(
                    accs::cuda::detail::BufCuda<TElem, TDim> & buf)
                -> accs::cuda::detail::BufCuda<TElem, TDim> &
                {
                    return buf;
                }
            };

            //#############################################################################
            //! The BufCuda native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPtrNative<
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf)
                -> TElem const *
                {
                    return buf.m_spMem.get();
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrNative(
                    accs::cuda::detail::BufCuda<TElem, TDim> & buf)
                -> TElem *
                {
                    return buf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The BufCuda pointer on device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPtrDev<
                accs::cuda::detail::BufCuda<TElem, TDim>,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrDev(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf,
                    accs::cuda::detail::DevCuda const & dev)
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
                    accs::cuda::detail::BufCuda<TElem, TDim> & buf,
                    accs::cuda::detail::DevCuda const & dev)
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
            //! The CUDA buffer pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                0u,
                accs::cuda::detail::BufCuda<TElem, TDim>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPitchBytes(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf)
                -> UInt
                {
                    return buf.m_uiPitchBytes;
                }
            };

            //#############################################################################
            //! The CUDA 1D memory allocation trait specialization.
            //#############################################################################
            template<
                typename T>
            struct Alloc<
                T,
                alpaka::dim::Dim1,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    accs::cuda::detail::DevCuda const & dev,
                    TExtents const & extents)
                -> accs::cuda::detail::BufCuda<T, alpaka::dim::Dim1>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const uiWidth(alpaka::extent::getWidth<UInt>(extents));
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
                        accs::cuda::detail::BufCuda<T, alpaka::dim::Dim1>(
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
                alpaka::dim::Dim2,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    accs::cuda::detail::DevCuda const & dev,
                    TExtents const & extents)
                -> accs::cuda::detail::BufCuda<T, alpaka::dim::Dim2>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    auto const uiWidth(alpaka::extent::getWidth<UInt>(extents));
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);
                    auto const uiHeight(alpaka::extent::getHeight<UInt>(extents));
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
                        accs::cuda::detail::BufCuda<T, alpaka::dim::Dim2>(
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
                alpaka::dim::Dim3,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                template<
                    typename TExtents>
                ALPAKA_FCT_HOST static auto alloc(
                    accs::cuda::detail::DevCuda const & dev,
                    TExtents const & extents)
                -> accs::cuda::detail::BufCuda<T, alpaka::dim::Dim3>
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            alpaka::extent::getWidth<UInt>(extents) * sizeof(T),
                            alpaka::extent::getHeight<UInt>(extents),
                            alpaka::extent::getDepth<UInt>(extents)));

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
                        << " ew: " << alpaka::extent::getWidth<UInt>(extents)
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
                        accs::cuda::detail::BufCuda<T, alpaka::dim::Dim3>(
                            dev,
                            reinterpret_cast<T *>(cudaPitchedPtrVal.ptr),
                            static_cast<UInt>(cudaPitchedPtrVal.pitch),
                            extents);
                }
            };

            //#############################################################################
            //! The CUDA memory mapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Map<
                accs::cuda::detail::BufCuda<TElem, TDim>,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto map(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf,
                    accs::cuda::detail::DevCuda const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        throw std::runtime_error("Mapping memory from one CUDA device into an other CUDA device not implemented!");
                    }
                    // If it is already the same device, nothing has to be mapped.
                }
            };

            //#############################################################################
            //! The CUDA memory unmapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Unmap<
                accs::cuda::detail::BufCuda<TElem, TDim>,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto unmap(
                    accs::cuda::detail::BufCuda<TElem, TDim> const & buf,
                    accs::cuda::detail::DevCuda const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        throw std::runtime_error("Unmapping memory from one CUDA device from an other CUDA device not implemented!");
                    }
                    // If it is already the same device, nothing has to be unmapped.
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace mem
        {
            //#############################################################################
            //! The BufCpu CUDA memory mapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Map<
                devs::cpu::detail::BufCpu<TElem, TDim>,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto map(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf,
                    accs::cuda::detail::DevCuda const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        // cudaHostRegisterMapped:
                        //   Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                        //   This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaHostRegister(
                                const_cast<void *>(reinterpret_cast<void const *>(alpaka::mem::getPtrNative(buf))),
                                alpaka::extent::getProductOfExtents<std::size_t>(buf) * sizeof(alpaka::mem::ElemT<devs::cpu::detail::BufCpu<TElem, TDim>>),
                                cudaHostRegisterMapped));
                    }
                    // If it is already the same device, nothing has to be mapped.
                }
            };

            //#############################################################################
            //! The BufCpu CUDA memory unmapping trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct Unmap<
                devs::cpu::detail::BufCpu<TElem, TDim>,
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto unmap(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf,
                    accs::cuda::detail::DevCuda const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(alpaka::dev::getDev(buf) != dev)
                    {
                        // Unmaps the memory range whose base address is specified by ptr, and makes it pageable again.
                        // TODO: If the memory has separately been pinned before we destroy the pinning state.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaHostUnregister(
                                const_cast<void *>(reinterpret_cast<void const *>(alpaka::mem::getPtrNative(buf)))));
                    }
                    // If it is already the same device, nothing has to be unmapped.
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
                accs::cuda::detail::DevCuda>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrDev(
                    devs::cpu::detail::BufCpu<TElem, TDim> const & buf,
                    accs::cuda::detail::DevCuda const & dev)
                -> TElem const *
                {
                    // TODO: Check if the memory is mapped at all!
                    TElem * pDev(nullptr);
                    ALPAKA_CUDA_RT_CHECK(
                        cudaHostGetDevicePointer(
                            &pDev,
                            const_cast<void *>(reinterpret_cast<void const *>(alpaka::mem::getPtrNative(buf))),
                            0));
                    return pDev;
                }
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static auto getPtrDev(
                    devs::cpu::detail::BufCpu<TElem, TDim> & buf,
                    accs::cuda::detail::DevCuda const & dev)
                -> TElem *
                {
                    // TODO: Check if the memory is mapped at all!
                    TElem * pDev(nullptr);
                    ALPAKA_CUDA_RT_CHECK(
                        cudaHostGetDevicePointer(
                            &pDev,
                            alpaka::mem::getPtrNative(buf),
                            0));
                    return pDev;
                }
            };
        }
    }
}
