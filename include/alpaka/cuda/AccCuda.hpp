/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/cuda/WorkSize.hpp>                 // TInterfacedWorkSize
#include <alpaka/cuda/Index.hpp>                    // TInterfacedIndex
#include <alpaka/cuda/Atomic.hpp>                   // TInterfacedAtomic

#include <alpaka/cuda/MemorySpace.hpp>              // MemorySpaceCuda
#include <alpaka/cuda/Memory.hpp>                   // MemCopy

#include <alpaka/cuda/Common.hpp>

#include <alpaka/interfaces/IAcc.hpp>               // IAcc
#include <alpaka/interfaces/KernelExecCreator.hpp>  // KernelExecCreator

#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>

#include <cstddef>                                  // std::size_t
#include <cstdint>                                  // unit8_t
#include <stdexcept>                                // std::except
#include <string>                                   // std::to_string
#include <sstream>                                  // std::stringstream
#ifdef _DEBUG
    #include <iostream>                             // std::cout
#endif

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! The CUDA kernel entry point.
            //-----------------------------------------------------------------------------
            template<typename TAcceleratedKernel, typename... TArgs>
            __global__ void cudaKernel(TAcceleratedKernel accedKernel, TArgs ... args)
            {
                accedKernel(std::forward<TArgs>(args)...);
            }

            template<typename TAcceleratedKernel>
            class KernelExecutor;

            //#############################################################################
            //! The base class for all CUDA accelerated kernels.
            //#############################################################################
            class AccCuda :
                protected TInterfacedWorkSize,
                private TInterfacedIndex,
                protected TInterfacedAtomic
            {
            public:
                using MemorySpace = MemorySpaceCuda;

                template<typename TAcceleratedKernel>
                friend class alpaka::cuda::detail::KernelExecutor;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AccCuda() :
                    TInterfacedWorkSize(),
                    TInterfacedIndex(),
                    TInterfacedAtomic()
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AccCuda(AccCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AccCuda(AccCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AccCuda & operator=(AccCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC ~AccCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in each dimension of a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static vec<3u> getSizeBlockKernelsMax()
                {
                    // TODO: CC < 2.0? Get from CUDA API.
                    return{1024u, 1024u, 64u};
                }
                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::uint32_t getSizeBlockKernelsLinearMax()
                {
                    // TODO: CC < 2.0? Get from CUDA API.
                    return 1024;
                }

                //-----------------------------------------------------------------------------
                //! Sets the CUDA device to use.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setDevice(int deviceNumber)
                {
#ifdef _DEBUG
                    std::cout << "[+] AccCuda::setDevice()" << std::endl;
#endif

                    int iNumGpus(0);
                    cudaGetDeviceCount(&iNumGpus);
                    if(iNumGpus < 1)
                    {
                        std::stringstream ssErr;
                        ssErr << "No CUDA capable devices detected!";

                        std::cerr << ssErr.str() << std::endl;
                        throw std::runtime_error(ssErr.str());
                    }
                    else if(iNumGpus < deviceNumber)
                    {
                        std::stringstream ssErr;
                        ssErr << "No CUDA device " << deviceNumber << " available, only " << iNumGpus << " devices found!";

                        std::cerr << ssErr.str() << std::endl;
                        throw std::runtime_error(ssErr.str());
                    }

                    cudaDeviceProp devProp;
                    ALPAKA_CUDA_CHECK(cudaGetDeviceProperties(&devProp, deviceNumber));
                    // Default compute mode (Multiple threads can use cudaSetDevice() with this device)
                    if(devProp.computeMode == cudaComputeModeDefault)
                    {
                        ALPAKA_CUDA_CHECK(cudaSetDevice(deviceNumber));
                        std::cout << "Set device to " << deviceNumber << ": " << devProp.name << std::endl;
#ifdef _DEBUG
                        std::size_t const uiKiB(1024);
                        std::size_t const uiMiB(uiKiB * uiKiB);
                        std::cout << "totalGlobalMem: " << devProp.totalGlobalMem/uiMiB << " MiB" << std::endl;
                        std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock/uiKiB << " KiB" << std::endl;
                        std::cout << "regsPerBlock: " << devProp.regsPerBlock << std::endl;
                        std::cout << "warpSize: " << devProp.warpSize << std::endl;
                        std::cout << "memPitch: " << devProp.memPitch << " B" << std::endl;
                        std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
                        std::cout << "maxThreadsDim[3]: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
                        std::cout << "maxGridSize[3]: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")" << std::endl;
                        std::cout << "clockRate: " << devProp.clockRate << " kHz" << std::endl;
                        std::cout << "totalConstMem: " << devProp.totalConstMem/uiKiB << " KiB" << std::endl;
                        std::cout << "major: " << devProp.major << std::endl;
                        std::cout << "minor: " << devProp.minor << std::endl;
                        std::cout << "textureAlignment: " << devProp.textureAlignment << std::endl;
                        std::cout << "texturePitchAlignment: " << devProp.texturePitchAlignment << std::endl;
                        //std::cout << "deviceOverlap: " << devProp.deviceOverlap << std::endl;    // Deprecated
                        std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
                        std::cout << "kernelExecTimeoutEnabled: " << devProp.kernelExecTimeoutEnabled << std::endl;
                        std::cout << "integrated: " << devProp.integrated << std::endl;
                        std::cout << "canMapHostMemory: " << devProp.canMapHostMemory << std::endl;
                        std::cout << "computeMode: " << devProp.computeMode << std::endl;
                        std::cout << "maxTexture1D: " << devProp.maxTexture1D << std::endl;
                        std::cout << "maxTexture1DLinear: " << devProp.maxTexture1DLinear << std::endl;
                        std::cout << "maxTexture2D[2]: " << devProp.maxTexture2D[0] << "x" << devProp.maxTexture2D[1] << std::endl;
                        std::cout << "maxTexture2DLinear[3]: " << devProp.maxTexture2DLinear[0] << "x" << devProp.maxTexture2DLinear[1] << "x" << devProp.maxTexture2DLinear[2] << std::endl;
                        std::cout << "maxTexture2DGather[2]: " << devProp.maxTexture2DGather[0] << "x" << devProp.maxTexture2DGather[1] << std::endl;
                        std::cout << "maxTexture3D[3]: " << devProp.maxTexture3D[0] << "x" << devProp.maxTexture3D[1] << "x" << devProp.maxTexture3D[2] << std::endl;
                        std::cout << "maxTextureCubemap: " << devProp.maxTextureCubemap << std::endl;
                        std::cout << "maxTexture1DLayered[2]: " << devProp.maxTexture1DLayered[0] << "x" << devProp.maxTexture1DLayered[1] << std::endl;
                        std::cout << "maxTexture2DLayered[3]: " << devProp.maxTexture2DLayered[0] << "x" << devProp.maxTexture2DLayered[1] << "x" << devProp.maxTexture2DLayered[2] << std::endl;
                        std::cout << "maxTextureCubemapLayered[2]: " << devProp.maxTextureCubemapLayered[0] << "x" << devProp.maxTextureCubemapLayered[1] << std::endl;
                        std::cout << "maxSurface1D: " << devProp.maxSurface1D << std::endl;
                        std::cout << "maxSurface2D[2]: " << devProp.maxSurface2D[0] << "x" << devProp.maxSurface2D[1] << std::endl;
                        std::cout << "maxSurface3D[3]: " << devProp.maxSurface3D[0] << "x" << devProp.maxSurface3D[1] << "x" << devProp.maxSurface3D[2] << std::endl;
                        std::cout << "maxSurface1DLayered[2]: " << devProp.maxSurface1DLayered[0] << "x" << devProp.maxSurface1DLayered[1] << std::endl;
                        std::cout << "maxSurface2DLayered[3]: " << devProp.maxSurface2DLayered[0] << "x" << devProp.maxSurface2DLayered[1] << "x" << devProp.maxSurface2DLayered[2] << std::endl;
                        std::cout << "maxSurfaceCubemap: " << devProp.maxSurfaceCubemap << std::endl;
                        std::cout << "maxSurfaceCubemapLayered[2]: " << devProp.maxSurfaceCubemapLayered[0] << "x" << devProp.maxSurfaceCubemapLayered[1] << std::endl;
                        std::cout << "surfaceAlignment: " << devProp.surfaceAlignment << std::endl;
                        std::cout << "concurrentKernels: " << devProp.concurrentKernels << std::endl;
                        std::cout << "ECCEnabled: " << devProp.ECCEnabled << std::endl;
                        std::cout << "pciBusID: " << devProp.pciBusID << std::endl;
                        std::cout << "pciDeviceID: " << devProp.pciDeviceID << std::endl;
                        std::cout << "pciDomainID: " << devProp.pciDomainID << std::endl;
                        std::cout << "tccDriver: " << devProp.tccDriver << std::endl;
                        std::cout << "asyncEngineCount: " << devProp.asyncEngineCount << std::endl;
                        std::cout << "unifiedAddressing: " << devProp.unifiedAddressing << std::endl;
                        std::cout << "memoryClockRate: " << devProp.memoryClockRate << " kHz" << std::endl;
                        std::cout << "memoryBusWidth: " << devProp.memoryBusWidth << " b" << std::endl;
                        std::cout << "l2CacheSize: " << devProp.l2CacheSize << " B" << std::endl;
                        std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
#endif
                    }
                    // Compute-exclusive-thread mode (Only one thread in one process will be able to use cudaSetDevice() with this device)
                    else if(devProp.computeMode == cudaComputeModeExclusive)
                    {
                        std::cerr << "Requested device is in computeMode cudaComputeModeExclusive.";
                        // TODO: Are we allowed to use the device in this case?
                    }
                    // Compute-prohibited mode (No threads can use cudaSetDevice() with this device)
                    else if(devProp.computeMode == cudaComputeModeProhibited)
                    {
                        std::cerr << "Requested device is in computeMode cudaComputeModeProhibited. It can not be selected!";
                    }
                    // Compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice() with this device)
                    else if(devProp.computeMode == cudaComputeModeExclusiveProcess)
                    {
                        std::cerr << "Requested device is in computeMode cudaComputeModeExclusiveProcess.";
                        // TODO: Are we allowed to use the device in this case?
                    }
                    else
                    {
                        std::cerr << "unknown computeMode!";
                    }

                    // Instruct CUDA to actively spin when waiting for results from the device.
                    // This can decrease latency when waiting for the device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.
                    ALPAKA_CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

#ifdef _DEBUG
                    std::cout << "[-] AccCuda::setDevice()" << std::endl;
#endif
                }

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
                ALPAKA_FCT_ACC typename alpaka::detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return this->TInterfacedIndex::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<TInterfacedWorkSize const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC void syncBlockKernels() const
                {
                    __syncthreads();
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T, std::size_t TuiNumElements>
                ALPAKA_FCT_ACC T * allocBlockSharedMem() const
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    __shared__ T shMem[TuiNumElements];
                    return &shMem;
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_ACC T * getBlockSharedExternMem() const
                {
                    extern __shared__ uint8_t shMem[];
                    return reinterpret_cast<T*>(shMem);
                }
            };

            //#############################################################################
            //! The executor for an accelerated serial kernel.
            //#############################################################################
            template<typename TAcceleratedKernel>
            class KernelExecutor :
                private TAcceleratedKernel
            {
                static_assert(std::is_base_of<IAcc<AccCuda>, TAcceleratedKernel>::value, "The TAcceleratedKernel for the cuda::detail::KernelExecutor has to inherit from IAcc<AccCuda>!");
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutor(TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...)
                {
#ifdef _DEBUG
                    std::cout << "[+] AccCuda::KernelExecutor()" << std::endl;
#endif
#ifdef _DEBUG
                    std::cout << "[-] AccCuda::KernelExecutor()" << std::endl;
#endif
                }
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor(KernelExecutor const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor(KernelExecutor &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor & operator=(KernelExecutor const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~KernelExecutor() noexcept = default;

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<typename TWorkSize, typename... TArgs>
                ALPAKA_FCT_HOST void operator()(IWorkSize<TWorkSize> const & workSize, TArgs && ... args) const
                {
#ifdef _DEBUG
                    std::cout << "[+] AccCuda::KernelExecutor::operator()" << std::endl;
#endif
                    auto const uiNumKernelsPerBlock(workSize.template getSize<Block, Kernels, Linear>());
                    auto const uiMaxKernelsPerBlock(AccCuda::getSizeBlockKernelsLinearMax());
                    if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                    {
                        throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the CUDA accelerator!").c_str());
                    }

                    auto const v3uiSizeGridBlocks(workSize.template getSize<Grid, Blocks, D3>());
                    auto const v3uiSizeBlockKernels(workSize.template getSize<Block, Kernels, D3>());
#ifdef _DEBUG
                    //std::cout << "GridBlocks: " << v3uiSizeGridBlocks << " BlockKernels: " << v3uiSizeBlockKernels<< std::endl;
#endif
                    dim3 gridDim(v3uiSizeGridBlocks[0], v3uiSizeGridBlocks[1], v3uiSizeGridBlocks[2]);
                    dim3 blockDim(v3uiSizeBlockKernels[0], v3uiSizeBlockKernels[1], v3uiSizeBlockKernels[2]);
#ifdef _DEBUG
                    //std::cout << "GridBlocks: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
                    //std::cout << "BlockKernels: (" <<  << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
#endif
                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(v3uiSizeBlockKernels, std::forward<TArgs>(args)...));

                    detail::cudaKernel<<<gridDim, blockDim, uiBlockSharedExternMemSizeBytes>>>(*static_cast<TAcceleratedKernel const *>(this), args...);
#ifdef _DEBUG
                    std::cout << "[-] AccCuda::KernelExecutor::operator()" << std::endl;
#endif
                }
            };
        }
    }

    using AccCuda = cuda::detail::AccCuda;

    //#############################################################################
    //! The specialization of the accelerator interface for CUDA.
    //#############################################################################
    template<>
    class IAcc<AccCuda> :
        protected AccCuda
    {
        using TAcc = AccCuda;
    public:
        //-----------------------------------------------------------------------------
        //! \return [The maximum number of memory sharing kernel executions | The maximum block size] allowed by the underlying accelerator.
        // TODO: Check if the used size is valid!
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST static vec<3u> getSizeBlockKernelsMax()
        {
            return TAcc::getSizeBlockKernelsMax();
        }
        //-----------------------------------------------------------------------------
        //! \return [The maximum number of memory sharing kernel executions | The maximum block size] allowed by the underlying accelerator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_HOST static std::uint32_t getSizeBlockKernelsLinearMax()
        {
            return TAcc::getSizeBlockKernelsLinearMax();
        }

        //-----------------------------------------------------------------------------
        //! \return The requested size.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_ACC typename detail::DimToRetType<TDimensionality>::type getSize() const
        {
            return TAcc::getSize<TOrigin, TUnit, TDimensionality>();
        }

    protected:
        //-----------------------------------------------------------------------------
        //! \return The requested index.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_ACC typename detail::DimToRetType<TDimensionality>::type getIdx() const
        {
            return TAcc::getIdx<TOrigin, TUnit, TDimensionality>();
        }

        //-----------------------------------------------------------------------------
        //! Execute the atomic operation on the given address with the given value.
        //! \return The old value before executing the atomic operation.
        //-----------------------------------------------------------------------------
        template<typename TOp, typename T>
        ALPAKA_FCT_ACC T atomicOp(T * const addr, T const & value) const
        {
            return TAcc::atomicOp<TOp, T>(addr, value);
        }

        //-----------------------------------------------------------------------------
        //! Syncs all kernels in the current block.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_ACC void syncBlockKernels() const
        {
            return TAcc::syncBlockKernels();
        }

        //-----------------------------------------------------------------------------
        //! \return Allocates block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T, std::size_t TuiNumElements>
        ALPAKA_FCT_ACC T * allocBlockSharedMem() const
        {
            static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

            return TAcc::allocBlockSharedMem<T, TuiNumElements>();
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the externally allocated block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T>
        ALPAKA_FCT_ACC T * getBlockSharedExternMem() const
        {
            return TAcc::getBlockSharedExternMem<T>();
        }
    };

    namespace detail
    {
        //#############################################################################
        //! The serial kernel executor builder.
        // TODO: How to assure that the kernel does not hold pointers to host memory?
        //#############################################################################
        template<typename TKernel, typename... TKernelConstrArgs>
        class KernelExecCreator<AccCuda, TKernel, TKernelConstrArgs...>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccCuda>::type;
            using TKernelExecutor = cuda::detail::KernelExecutor<TAcceleratedKernel>;

            // Copying a kernel onto the CUDA device has some extra requirements of being trivially copyable:
            // A trivially copyable class is a class that
            // 1. Has no non-trivial copy constructors(this also requires no virtual functions or virtual bases)
            // 2. Has no non-trivial move constructors
            // 3. Has no non-trivial copy assignment operators
            // 4. Has no non-trivial move assignment operators
            // 5. Has a trivial destructor
            //
#ifndef __GNUC__    // FIXME: Find out which version > 4.8.0 does support the std::is_trivially_copyable
            // TODO: is_standard_layout is even stricter. Is is_trivially_copyable enough?
            static_assert(std::is_trivially_copyable<TAcceleratedKernel>::value, "The given kernel functor has to be trivially copyable to be used on a CUDA device!");
#endif
            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST TKernelExecutor operator()(TKernelConstrArgs && ... args) const
            {
                return TKernelExecutor(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
