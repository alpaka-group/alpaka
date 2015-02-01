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
#include <alpaka/cuda/Mem.hpp>                      // MemCopy
#include <alpaka/cuda/Stream.hpp>                   // StreamCuda
#include <alpaka/cuda/Event.hpp>                    // EventCuda
#include <alpaka/cuda/StreamEventTraits.hpp>        // StreamCuda & EventCuda
#include <alpaka/cuda/Device.hpp>                   // Devices

// Specialized traits.
#include <alpaka/core/KernelExecCreator.hpp>        // KernelExecCreator
#include <alpaka/traits/Acc.hpp>                    // GetAcc

// Implementation details.
#include <alpaka/cuda/Common.hpp>
#include <alpaka/interfaces/IAcc.hpp>               // IAcc
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>

#include <cstddef>                                  // std::size_t
#include <cstdint>                                  // std::uint32_t
#include <stdexcept>                                // std::except
#include <string>                                   // std::to_string
#include <utility>                                  // std::forward
#include <tuple>                                    // std::tuple

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

// Workarounds.
#include <boost/predef.h>

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            template<
                typename TAcceleratedKernel>
            class KernelExecutorCuda;

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
                using MemSpace = mem::MemSpaceCuda;

                template<
                    typename TAcceleratedKernel>
                friend class cuda::detail::KernelExecutorCuda;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda() :
                    WorkDivCuda(),
                    IdxCuda(),
                    AtomicCuda()
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda & operator=(AccCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY /*virtual*/ ~AccCuda() noexcept = default;

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested extents.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin,
                    typename TUnit,
                    typename TDimensionality = dim::Dim3>
                ALPAKA_FCT_ACC_CUDA_ONLY typename dim::DimToVecT<TDimensionality> getWorkDiv() const
                {
                    return workdiv::getWorkDiv<TOrigin, TUnit, TDimensionality>(
                        *static_cast<WorkDivCuda const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! \return The requested indices.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin, 
                    typename TUnit, 
                    typename TDimensionality = dim::Dim3>
                ALPAKA_FCT_ACC_CUDA_ONLY typename dim::DimToVecT<TDimensionality> getIdx() const
                {
                    return idx::getIdx<TOrigin, TUnit, TDimensionality>(
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
                ALPAKA_FCT_ACC T atomicOp(
                    T * const addr,
                    T const & value) const
                {
                    return atomic::atomicOp<TOp, T>(
                        addr,
                        value,
                        *static_cast<AtomicCuda const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY void syncBlockKernels() const
                {
                    __syncthreads();
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T, 
                    std::size_t TuiNumElements>
                ALPAKA_FCT_ACC_CUDA_ONLY T * allocBlockSharedMem() const
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
                ALPAKA_FCT_ACC_CUDA_ONLY T * getBlockSharedExternMem() const
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
        }
    }
    
    //#############################################################################
    //! The CUDA accelerator interface specialization.
    //!
    //! This specialization is required because the functions are not allowed to be declared ALPAKA_FCT_ACC but are required to be ALPAKA_FCT_ACC_CUDA_ONLY only.
    //#############################################################################
    template<>
    class IAcc<
        AccCuda> :
            protected AccCuda
    {
    public:
        //-----------------------------------------------------------------------------
        //! \return The requested extents.
        //-----------------------------------------------------------------------------
        template<
            typename TOrigin, 
            typename TUnit, 
            typename TDimensionality = dim::Dim3>
        ALPAKA_FCT_ACC_CUDA_ONLY typename dim::DimToVecT<TDimensionality> getWorkDiv() const
        {
            return AccCuda::getWorkDiv<TOrigin, TUnit, TDimensionality>();
        }

        //-----------------------------------------------------------------------------
        //! \return The requested indices.
        //-----------------------------------------------------------------------------
        template<
            typename TOrigin, 
            typename TUnit, 
            typename TDimensionality = dim::Dim3>
        ALPAKA_FCT_ACC_CUDA_ONLY typename dim::DimToVecT<TDimensionality> getIdx() const
        {
            return AccCuda::getIdx<TOrigin, TUnit, TDimensionality>();
        }

        //-----------------------------------------------------------------------------
        //! Execute the atomic operation on the given address with the given value.
        //! \return The old value before executing the atomic operation.
        //-----------------------------------------------------------------------------
        template<
            typename TOp, 
            typename T>
        ALPAKA_FCT_ACC_CUDA_ONLY T atomicOp(
            T * const addr, 
            T const & value) const
        {
            return AccCuda::atomicOp<TOp, T>(addr, value);
        }

        //-----------------------------------------------------------------------------
        //! Syncs all kernels in the current block.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_ACC_CUDA_ONLY void syncBlockKernels() const
        {
            return AccCuda::syncBlockKernels();
        }

        //-----------------------------------------------------------------------------
        //! \return Allocates block shared memory.
        //-----------------------------------------------------------------------------
        template<
            typename T, 
            std::size_t TuiNumElements>
        ALPAKA_FCT_ACC_CUDA_ONLY T * allocBlockSharedMem() const
        {
            static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

            return AccCuda::allocBlockSharedMem<T, TuiNumElements>();
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the externally allocated block shared memory.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_ACC_CUDA_ONLY T * getBlockSharedExternMem() const
        {
            return AccCuda::getBlockSharedExternMem<T>();
        }
    };
    
    namespace detail
    {
        //-----------------------------------------------------------------------------
        //! Holds a value of true if the given number is a power of two, false else.
        //-----------------------------------------------------------------------------
        template<
            std::size_t TuiVal>
        struct IsPowerOfTwo : 
            std::integral_constant<bool, ((TuiVal != 0) && ((TuiVal & (~TuiVal + 1)) == TuiVal))>
        {};

        //-----------------------------------------------------------------------------
        //! Holds a value with the pointer aligned by rounding upwards.
        //-----------------------------------------------------------------------------
        template<
            std::uintptr_t TuiAddress,
            std::size_t TuiAlignment>
        struct AlignUp : 
            std::integral_constant<std::uintptr_t, (TuiAddress + (TuiAlignment-1)) & ~(TuiAlignment-1)>
        {
            static_assert(TuiAlignment > 0, "The given alignment has to be greater zero!");
            static_assert(IsPowerOfTwo<TuiAlignment>::value, "The given alignment has to be a power of two!");
        };
    }

    namespace cuda
    {
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! The CUDA kernel entry point.
            //-----------------------------------------------------------------------------
            template<
                typename TAcceleratedKernel,
                typename... TArgs>
            __global__ void cudaKernel(
                TAcceleratedKernel accedKernel,
                TArgs ... args)
            {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error "Cuda device capability >= 2.0 is required!"
#endif
                IAcc<AccCuda> acc;

                accedKernel(
                    acc,
                    std::forward<TArgs>(args)...);
            }

            //#############################################################################
            //! The CUDA accelerator executor.
            //#############################################################################
            template<
                typename TAcceleratedKernel>
            class KernelExecutorCuda :
                private TAcceleratedKernel/*,
                private IAcc<AccCuda>*/
            {
#if (!BOOST_COMP_GNUC) || (BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(5, 0, 0))
                static_assert(std::is_trivially_copyable<TAcceleratedKernel>::value, "The given kernel functor has to fulfill is_trivially_copyable!");
#endif

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv, 
                    typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutorCuda(
                    TWorkDiv const & workDiv, 
                    StreamCuda const & stream, 
                    TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...),
                    m_Stream(stream)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    m_v3uiGridBlocksExtents = workdiv::getWorkDiv<Grid, Blocks, dim::Dim3>(workDiv);
                    m_v3uiBlockKernelsExtents = workdiv::getWorkDiv<Block, Kernels, dim::Dim3>(workDiv);

                    // TODO: Check that (sizeof(TAcceleratedKernel) * m_v3uiBlockKernelsExtents.prod()) < available memory size
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorCuda(KernelExecutorCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorCuda(KernelExecutorCuda &&) = default;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorCuda & operator=(KernelExecutorCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                ALPAKA_FCT_HOST virtual ~KernelExecutorCuda() = default;
#else
                ALPAKA_FCT_HOST virtual ~KernelExecutorCuda() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<
                    typename... TArgs>
                ALPAKA_FCT_HOST void operator()(
                    TArgs && ... args) const
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::size_t uiPrintfFifoSize;
                    cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << "uiPrintfFifoSize: " << uiPrintfFifoSize << std::endl;
                    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, uiPrintfFifoSize*100);
                    //cudaDeviceGetLimit(&uiPrintfFifoSize, cudaLimitPrintfFifoSize);
                    //std::cout << "uiPrintfFifoSize: " <<  uiPrintfFifoSize << std::endl;
#endif

                    dim3 gridDim(
                        static_cast<unsigned int>(m_v3uiGridBlocksExtents[0u]), 
                        static_cast<unsigned int>(m_v3uiGridBlocksExtents[1u]), 
                        static_cast<unsigned int>(m_v3uiGridBlocksExtents[2u]));
                    dim3 blockDim(
                        static_cast<unsigned int>(m_v3uiBlockKernelsExtents[0u]), 
                        static_cast<unsigned int>(m_v3uiBlockKernelsExtents[1u]), 
                        static_cast<unsigned int>(m_v3uiBlockKernelsExtents[2u]));
                    
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << "v3uiBlockKernelsExtents: " <<  gridDim.x << " " <<  gridDim.y << " " <<  gridDim.z << std::endl;
                    std::cout << "v3uiBlockKernelsExtents: " <<  blockDim.x << " " <<  blockDim.y << " " <<  blockDim.z << std::endl;
#endif

                    // Get the size of the block shared extern memory.
                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(
                        m_v3uiBlockKernelsExtents, 
                        std::forward<TArgs>(args)...));

                    // \TODO: The following block should be in a lock.
                    {
                        // Set the environment settings of the following cuda kernel invocation.
                        ALPAKA_CUDA_CHECK(cudaConfigureCall(
                            gridDim,
                            blockDim,
                            uiBlockSharedExternMemSizeBytes,
                            *m_Stream.m_spCudaStream.get()));
                        // Push the arguments.
                        pushCudaKernelArgument<0>(
                            *static_cast<TAcceleratedKernel const *>(this),
                            std::forward<TArgs>(args)...);
                        // And execute it.
                        ALPAKA_CUDA_CHECK(cudaLaunch(
                            cudaKernel<TAcceleratedKernel, TArgs...>));

                        /*cudaKernel<TAcceleratedKernel, TArgs...><<<
                            gridDim,
                            blockDim,
                            uiBlockSharedExternMemSizeBytes,
                            *m_Stream.m_spCudaStream.get()>>>(
                                *static_cast<TAcceleratedKernel const *>(this),
                                std::forward<TArgs>(args)...);*/
                    }
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // Wait for the kernel execution to finish but do not check error return of this call.
                    // Do not use the alpaka::wait method because it checks the error itself but we want to give a custom error message.
                    cudaStreamSynchronize(*m_Stream.m_spCudaStream.get());
                    cudaDeviceSynchronize();
                    cudaError_t const error(cudaGetLastError());
                    if(error != cudaSuccess)
                    {
                        std::string const sError("The execution of kernel '" + std::string(typeid(TAcceleratedKernel).name()) + " failed with error: '" + std::string(cudaGetErrorString(error)) + "'"); \
                        std::cerr << sError << std::endl;\
                        throw std::runtime_error(sError);\
                    }
#endif
                }

            private:
                //-----------------------------------------------------------------------------
                //! Push kernel arguments.
                //-----------------------------------------------------------------------------
                template<
                    std::size_t TuiOffset>
                void pushCudaKernelArgument() const
                {
                    // The base case to push zero arguments.
                }
                //-----------------------------------------------------------------------------
                //! Push kernel arguments.
                //-----------------------------------------------------------------------------
                template<
                    std::size_t TuiOffset, 
                    typename T0, 
                    typename... TArgs>
                void pushCudaKernelArgument(
                    T0 && arg0, 
                    TArgs && ... args) const
                {
                    // Push the first argument.
                    ALPAKA_CUDA_CHECK(cudaSetupArgument(
                        &arg0, 
                        sizeof(typename std::decay<T0>::type), 
                        alpaka::detail::AlignUp<TuiOffset, ALPAKA_ALIGNOF(typename std::decay<T0>::type)>::value));

                    // Push the rest of the arguments recursively.
                    pushCudaKernelArgument<
                        alpaka::detail::AlignUp<TuiOffset, ALPAKA_ALIGNOF(typename std::decay<T0>::type)>::value
                        + sizeof(typename std::decay<T0>::type)>(
                            std::forward<TArgs>(args)...);
                }

            private:
                Vec<3u> m_v3uiGridBlocksExtents;
                Vec<3u> m_v3uiBlockKernelsExtents;

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
            template<
                typename AcceleratedKernel>
            struct GetAcc<
                cuda::detail::KernelExecutorCuda<AcceleratedKernel>>
            {
                using type = AccCuda;
            };
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The CUDA accelerator kernel executor builder.
        // \TODO: How to assure that the kernel does not hold pointers to host memory?
        //#############################################################################
        template<
            typename TKernel, 
            typename... TKernelConstrArgs>
        class KernelExecCreator<
            AccCuda, 
            TKernel, 
            TKernelConstrArgs...>
        {
        public:
            using AcceleratedKernel = typename boost::mpl::apply<TKernel, AccCuda>::type;
            using AcceleratedKernelExecutorExtent = KernelExecutorExtent<cuda::detail::KernelExecutorCuda<AcceleratedKernel>, TKernelConstrArgs...>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST AcceleratedKernelExecutorExtent operator()(
                TKernelConstrArgs && ... args) const
            {
                return AcceleratedKernelExecutorExtent(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
