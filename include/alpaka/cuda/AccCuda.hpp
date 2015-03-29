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

// Base classes.
#include <alpaka/cuda/AccCudaFwd.hpp>
#include <alpaka/cuda/WorkDiv.hpp>                  // WorkDivCuda
#include <alpaka/cuda/Idx.hpp>                      // IdxCuda
#include <alpaka/cuda/Atomic.hpp>                   // AtomicCuda

// User functionality.
#include <alpaka/cuda/Mem.hpp>                      // Copy
#include <alpaka/cuda/Stream.hpp>                   // StreamCuda
#include <alpaka/cuda/Event.hpp>                    // EventCuda
#include <alpaka/cuda/StreamEventTraits.hpp>        // StreamCuda & EventCuda
#include <alpaka/cuda/Device.hpp>                   // Devices

// Specialized traits.
#include <alpaka/traits/Acc.hpp>                    // AccType
#include <alpaka/traits/Exec.hpp>                   // ExecType

// Implementation details.
#include <alpaka/cuda/Common.hpp>
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>

#include <boost/predef.h>                           // workarounds

#include <cstdint>                                  // std::uint32_t
#include <stdexcept>                                // std::runtime_error
#include <string>                                   // std::to_string
#include <utility>                                  // std::forward
#include <tuple>                                    // std::tuple

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            // Forward declarations.
            /*template<
                typename TKernelFunctor,
                typename... TArgs>
            __global__ void cudaKernel(
                TKernelFunctor kernelFunctor,
                TArgs ... args);*/

            //class KernelExecCuda;

            //#############################################################################
            //! The CUDA accelerator.
            //!
            //! This accelerator allows parallel kernel execution on devices supporting CUDA.
            //#############################################################################
            class AccCuda :
                protected WorkDivCuda,
                private IdxCuda,
                protected AtomicCuda
            {
            public:
                using MemSpace = mem::SpaceCuda;
                
                /*template<
                    typename TKernelFunctor,
                    typename... TArgs>
                friend void ::alpaka::cuda::detail::cudaKernel(
                    TKernelFunctor kernelFunctor,
                    TArgs ... args);*/

                //friend class ::alpaka::cuda::detail::KernelExecCuda;
                
            //private:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda() :
                    WorkDivCuda(),
                    IdxCuda(),
                    AtomicCuda()
                {}

            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda &&) = delete;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY auto operator=(AccCuda const &) -> AccCuda & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~AccCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The requested extents.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin,
                    typename TUnit,
                    typename TDim = dim::Dim3>
                ALPAKA_FCT_ACC_CUDA_ONLY auto getWorkDiv() const
                -> DimToVecT<TDim>
                {
                    return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                        *static_cast<WorkDivCuda const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! \return The requested indices.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin, 
                    typename TUnit, 
                    typename TDim = dim::Dim3>
                ALPAKA_FCT_ACC_CUDA_ONLY auto getIdx() const
                -> DimToVecT<TDim>
                {
                    return idx::getIdx<TOrigin, TUnit, TDim>(
                        *static_cast<IdxCuda const *>(this),
                        *static_cast<WorkDivCuda const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Execute the atomic operation on the given address with the given value.
                //! \return The old value before executing the atomic operation.
                //-----------------------------------------------------------------------------
                template<
                    typename TOp,
                    typename T>
                ALPAKA_FCT_ACC auto atomicOp(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    return atomic::atomicOp<TOp, T>(
                        addr,
                        value,
                        *static_cast<AtomicCuda const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all threads in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY auto syncBlockThreads() const
                -> void
                {
                    __syncthreads();
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T, 
                    UInt TuiNumElements>
                ALPAKA_FCT_ACC_CUDA_ONLY auto allocBlockSharedMem() const
                -> T *
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    __shared__ T shMem[TuiNumElements];
                    return shMem;
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T>
                ALPAKA_FCT_ACC_CUDA_ONLY auto getBlockSharedExternMem() const
                -> T *
                {
                    // Because unaligned access to variables is not allowed in device code, 
                    // we have to use the widest possible type to have all types aligned correctly. 
                    // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
                    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#vector-types
                    extern __shared__ float4 shMem[];
                    //printf("s %p\n", shMem);
                    return reinterpret_cast<T *>(shMem);
                }
            };

            //-----------------------------------------------------------------------------
            //! The CUDA kernel entry point.
            // \NOTE: A __global__ function or function template cannot have a trailing return type.
            //-----------------------------------------------------------------------------
            template<
                typename TKernelFunctor,
                typename... TArgs>
            __global__ void cudaKernel(
                TKernelFunctor kernelFunctor,
                TArgs ... args)
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error "Cuda device capability >= 2.0 is required!"
#endif
                AccCuda acc;

                kernelFunctor(
                    acc,
                    args...);
            }

            //#############################################################################
            //! The CUDA accelerator executor.
            //#############################################################################
            class KernelExecCuda
            {
#if (!__GLIBCXX__) // libstdc++ even for gcc-4.9 does not support std::is_trivially_copyable.
                static_assert(std::is_trivially_copyable<TKernelFunctor>::value, "The given kernel functor has to fulfill is_trivially_copyable!");
#endif

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_HOST KernelExecCuda(
                    TWorkDiv const & workDiv, 
                    StreamCuda const & stream) :
                        m_Stream(stream),
                        m_v3uiGridBlockExtents(workdiv::getWorkDiv<Grid, Blocks, dim::Dim3>(workDiv)),
                        m_v3uiBlockThreadExtents(workdiv::getWorkDiv<Block, Threads, dim::Dim3>(workDiv))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecCuda(KernelExecCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecCuda(KernelExecCuda &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(KernelExecCuda const &) -> KernelExecCuda & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                ALPAKA_FCT_HOST virtual ~KernelExecCuda() = default;
#else
                ALPAKA_FCT_HOST virtual ~KernelExecCuda() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Executes the kernel functor.
                //-----------------------------------------------------------------------------
                template<
                    typename TKernelFunctor,
                    typename... TArgs>
                ALPAKA_FCT_HOST auto operator()(
                    // \NOTE: No universal reference (&&) or const reference (const &) is allowed as the parameter type because the kernel launch language extension expects the arguments by value.
                    // This forces the type of a float argument given with std::forward to this function to be of type float instead of e.g. "float const & __ptr64" (MSVC).
                    // If not given by value, the kernel launch code does not copy the value but the pointer to the value location.
                    TKernelFunctor kernelFunctor,
                    TArgs ... args) const
                -> void
                {
#if (!__GLIBCXX__) // libstdc++ even for gcc-4.9 does not support std::is_trivially_copyable.
                    static_assert(std::is_trivially_copyable<TKernelFunctor>::value, "The given kernel functor has to fulfill is_trivially_copyable!");
#endif
                    // TODO: Check that (sizeof(TKernelFunctor) * m_v3uiBlockThreadExtents.prod()) < available memory size

                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //std::size_t uiPrintfFifoSize;
                    //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << "uiPrintfFifoSize: " << uiPrintfFifoSize << std::endl;
                    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, uiPrintfFifoSize*10);
                    //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << "uiPrintfFifoSize: " <<  uiPrintfFifoSize << std::endl;
#endif

                    dim3 const gridDim(
                        static_cast<unsigned int>(m_v3uiGridBlockExtents[0u]), 
                        static_cast<unsigned int>(m_v3uiGridBlockExtents[1u]), 
                        static_cast<unsigned int>(m_v3uiGridBlockExtents[2u]));
                    dim3 const blockDim(
                        static_cast<unsigned int>(m_v3uiBlockThreadExtents[0u]), 
                        static_cast<unsigned int>(m_v3uiBlockThreadExtents[1u]), 
                        static_cast<unsigned int>(m_v3uiBlockThreadExtents[2u]));
                    
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << "v3uiBlockThreadExtents: " <<  gridDim.x << " " <<  gridDim.y << " " <<  gridDim.z << std::endl;
                    std::cout << "v3uiBlockThreadExtents: " <<  blockDim.x << " " <<  blockDim.y << " " <<  blockDim.z << std::endl;
#endif

                    // Get the size of the block shared extern memory.
                    auto const uiBlockSharedExternMemSizeBytes(getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccCuda>(
                        m_v3uiBlockThreadExtents, 
                        std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                        << std::endl;
#endif

                    // \TODO: The following block should be in a lock.
                    {
                        cudaKernel<TKernelFunctor, TArgs...><<<
                            gridDim,
                            blockDim,
                            uiBlockSharedExternMemSizeBytes,
                            *m_Stream.m_spCudaStream.get()>>>(
                                kernelFunctor,
                                args...);
                    }
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(*m_Stream.m_spCudaStream.get());
                    //cudaDeviceSynchronize();
                    cudaError_t const error(cudaGetLastError());
                    if(error != cudaSuccess)
                    {
                        std::string const sError("The execution of kernel '" + std::string(typeid(TKernelFunctor).name()) + " failed with error: '" + std::string(cudaGetErrorString(error)) + "'");
                        std::cerr << sError << std::endl;
                        ALPAKA_DEBUG_BREAK;
                        throw std::runtime_error(sError);
                    }
#endif
                }

            private:
                Vec<3u> const m_v3uiGridBlockExtents;
                Vec<3u> const m_v3uiBlockThreadExtents;

            public:
                StreamCuda m_Stream;
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CUDA accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                cuda::detail::KernelExecCuda>
            {
                using type = AccCuda;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The CUDA accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                AccCuda>
            {
                using type = cuda::detail::KernelExecCuda;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CUDA accelerator kernel executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                cuda::detail::KernelExecCuda>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    cuda::detail::KernelExecCuda const & exec)
                -> cuda::detail::StreamCuda
                {
                    return exec.m_Stream;
                }
            };
        }
    }
}
