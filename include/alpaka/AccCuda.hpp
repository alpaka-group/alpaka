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

#include <alpaka/KernelExecutorBuilder.hpp> // KernelExecutorBuilder
#include <alpaka/WorkSize.hpp>              // IWorkSize, WorkSizeDefault
#include <alpaka/Index.hpp>                 // IIndex
#include <alpaka/Atomic.hpp>                // IAtomic

#include <alpaka/IAcc.hpp>                  // IAcc

#include <cstddef>                          // std::size_t
#include <cstdint>                          // unit8_t
#include <stdexcept>                        // std::except
#include <string>                           // std::to_string
#include <sstream>                          // std::stringstream
#ifdef _DEBUG
    #include <iostream>                     // std::cout
#endif

#include <boost/mpl/apply.hpp>              // boost::mpl::apply

#include <cuda.h>

#define ALPAKA_CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){std::cerr<<"<"<<__FILE__<<">:"<<__LINE__<<std::endl; throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));}}

#define ALPAKA_CUDA_CHECK_MSG(cmd,msg) {cudaError_t error = cmd; if(error!=cudaSuccess){std::cerr<<"<"<<__FILE__<<">:"<<__LINE__<<msg<<std::endl; throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));}}

#define ALPAKA_CUDA_CHECK_NO_EXCEP(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("[CUDA] Error: <%s>:%i ",__FILE__,__LINE__);}}

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! This class that holds the implementation details for the work sizes of the CUDA accelerator.
            //#############################################################################
            class WorkSizeCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA WorkSizeCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA WorkSizeCuda(WorkSizeCuda const & other) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA WorkSizeCuda(WorkSizeCuda && other) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA WorkSizeCuda & operator=(WorkSizeCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU ~WorkSizeCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The grid dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA vec<3u> getSizeGridBlocks() const
                {
//#ifdef __CUDA_ARCH__
                    return {gridDim.x, gridDim.y, gridDim.z};
//#else
//                    throw std::logic_error("WorkSizeCuda can not be used in non-CUDA Code!");
//#endif
                }
                //-----------------------------------------------------------------------------
                //! \return The block dimensions of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA vec<3u> getSizeBlockKernels() const
                {
//#ifdef __CUDA_ARCH__
                    return {blockDim.x, blockDim.y, blockDim.z};
//#else
//                    throw std::logic_error("WorkSizeCuda can not be used in non-CUDA Code!");
//#endif
                }
            };
            using TInterfacedWorkSize = alpaka::IWorkSize<WorkSizeCuda>;

            //#############################################################################
            //! This class that holds the implementation details for the indexing of the CUDA accelerator.
            //#############################################################################
            class IndexCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA IndexCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA IndexCuda(IndexCuda const & other) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA IndexCuda(IndexCuda && other) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA IndexCuda & operator=(IndexCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU ~IndexCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA vec<3u> getIdxBlockKernel() const
                {
                    return {threadIdx.x, threadIdx.y, threadIdx.z};
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA vec<3u> getIdxGridBlock() const
                {
                    return {blockIdx.x, blockIdx.y, blockIdx.z};
                }
            };
            using TInterfacedIndex = alpaka::detail::IIndex<IndexCuda>;
            
            //#############################################################################
            //! This class that holds the implementation details for the atomic operations of the CUDA accelerator.
            //#############################################################################
            class AtomicCuda
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AtomicCuda() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AtomicCuda(AtomicCuda const & other) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AtomicCuda(AtomicCuda && other) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AtomicCuda & operator=(AtomicCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU ~AtomicCuda() noexcept = default;
            };
            using TInterfacedAtomic = alpaka::detail::IAtomic<AtomicCuda>;
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The specialization to execute the requested atomic operation of the CUDA accelerator.
        //#############################################################################
        // TODO: These ops are only implemented for int and unsigned int; additionally Exch and Add (2.0+) for float; some also for unsigned long long int
        // See: http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions how to implement everything with CAS
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Add, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicAdd(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Sub, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicSub(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Min, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicMin(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Max, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicMax(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Exch, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicExch(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Inc, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicInc(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Dec, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicDec(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, And, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicAnd(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Or, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicOr(addr, value);
            }
        };
        template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Xor, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & value)
            {
                return atomicXOr(addr, value);
            }
        };
        /*template<typename T>
        struct AtomicOp<cuda::detail::AtomicCuda, Cas, T>
        {
            ALPAKA_FCT_CUDA static T atomicOp(cuda::detail::AtomicCuda const & , T * const addr, T const & compare, T const & value)
            {
                return atomicCAS(addr, compare, value);
            }
        };*/
    }

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

            //#############################################################################
            //! The base class for all CUDA accelerated kernels.
            //#############################################################################
            class AccCuda :
                protected TInterfacedWorkSize,
                private TInterfacedIndex,
                protected TInterfacedAtomic
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AccCuda() :
                    TInterfacedWorkSize(),
                    TInterfacedIndex(),
                    TInterfacedAtomic()
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AccCuda(AccCuda const & other) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AccCuda(AccCuda && other) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA AccCuda & operator=(AccCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU ~AccCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in each dimension of a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU static vec<3u> getSizeBlockKernelsMax()
                {
                    // TODO: CC < 2.0? Get from CUDA API.
                    return{1024u, 1024u, 64u};
                }
                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU static std::uint32_t getSizeBlockKernelsLinearMax()
                {
                    // TODO: CC < 2.0? Get from CUDA API.
                    return 1024;
                }

                //-----------------------------------------------------------------------------
                //! Sets the CUDA device to use.
                //-----------------------------------------------------------------------------
                static void setDevice(int deviceNumber)
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
                        std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock/uiMiB << " MiB" << std::endl;
                        std::cout << "regsPerBlock: " << devProp.regsPerBlock << std::endl;
                        std::cout << "warpSize: " << devProp.warpSize << std::endl;
                        std::cout << "memPitch: " << devProp.memPitch << " B" << std::endl;
                        std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
                        std::cout << "maxThreadsDim[3]: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
                        std::cout << "maxGridSize[3]: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")" << std::endl;
                        std::cout << "clockRate: " << devProp.clockRate << " kHz" << std::endl;
                        std::cout << "totalConstMem: " << devProp.totalConstMem << " B" << std::endl;
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
                ALPAKA_FCT_CUDA typename detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return this->TInterfacedIndex::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<TInterfacedWorkSize const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CUDA void syncBlockKernels() const
                {
                    __syncthreads();
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T, std::size_t UiNumElements>
                ALPAKA_FCT_CUDA T * allocBlockSharedMem() const
                {
                    static_assert(UiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    __shared__ T shMem[UiNumElements];
                    return &shMem;
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_CUDA T * getBlockSharedExternMem() const
                {
                    extern __shared__ uint8_t shMem[];
                    return reinterpret_cast<T*>(shMem);
                }

            public:
                //#############################################################################
                //! The executor for an accelerated serial kernel.
                // TODO: Check that TAcceleratedKernel inherits from the correct accelerator.
                //#############################################################################
                template<typename TAcceleratedKernel>
                class KernelExecutor :
                    protected TAcceleratedKernel
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<typename... TKernelConstrArgs>
                    ALPAKA_FCT_CPU KernelExecutor(TKernelConstrArgs && ... args) :
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
                    ALPAKA_FCT_CPU KernelExecutor(KernelExecutor const & other) = default;
                    //-----------------------------------------------------------------------------
                    //! Move-constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_CPU KernelExecutor(KernelExecutor && other) = default;
                    //-----------------------------------------------------------------------------
                    //! Assignment-operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_CPU KernelExecutor & operator=(KernelExecutor const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_CPU ~KernelExecutor() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! Executes the accelerated kernel.
                    //-----------------------------------------------------------------------------
                    template<typename TWorkSize, typename... TArgs>
                    ALPAKA_FCT_CPU void operator()(IWorkSize<TWorkSize> const & workSize, TArgs && ... args) const
                    {
#ifdef _DEBUG
                        std::cout << "[+] AccCuda::KernelExecutor::operator()" << std::endl;
#endif
                        auto const uiNumKernelsPerBlock(workSize.getSize<Block, Kernels, Linear>());
                        auto const uiMaxKernelsPerBlock(this->TAcceleratedKernel::getSizeBlockKernelsLinearMax());
                        if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                        {
                            throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the CUDA accelerator!").c_str());
                        }

                        auto const v3uiSizeGridBlocks(workSize.getSize<Grid, Blocks, D3>());
                        auto const v3uiSizeBlockKernels(workSize.getSize<Block, Kernels, D3>());
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
        ALPAKA_FCT_CPU static vec<3u> getSizeBlockKernelsMax()
        {
            return TAcc::getSizeBlockKernelsMax();
        }
        //-----------------------------------------------------------------------------
        //! \return [The maximum number of memory sharing kernel executions | The maximum block size] allowed by the underlying accelerator.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_CPU static std::uint32_t getSizeBlockKernelsLinearMax()
        {
            return TAcc::getSizeBlockKernelsLinearMax();
        }

        //-----------------------------------------------------------------------------
        //! \return The requested size.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_CUDA typename detail::DimToRetType<TDimensionality>::type getSize() const
        {
            return TAcc::getSize<TOrigin, TUnit, TDimensionality>();
        }

    protected:
        //-----------------------------------------------------------------------------
        //! \return The requested index.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_CUDA typename detail::DimToRetType<TDimensionality>::type getIdx() const
        {
            return TAcc::getIdx<TOrigin, TUnit, TDimensionality>();
        }

        //-----------------------------------------------------------------------------
        //! Execute the atomic operation on the given address with the given value.
        //! \return The old value before executing the atomic operation.
        //-----------------------------------------------------------------------------
        template<typename TOp, typename T>
        ALPAKA_FCT_CUDA T atomicOp(T * const addr, T const & value) const
        {
            return TAcc::atomicOp<TOp, T>(addr, value);
        }

        //-----------------------------------------------------------------------------
        //! Syncs all kernels in the current block.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_CUDA void syncBlockKernels() const
        {
            return TAcc::syncBlockKernels();
        }

        //-----------------------------------------------------------------------------
        //! \return Allocates block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T, std::size_t UiNumElements>
        ALPAKA_FCT_CUDA T * allocBlockSharedMem() const
        {
            static_assert(UiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

            return TAcc::allocBlockSharedMem<T, UiNumElements>();
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the externally allocated block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T>
        ALPAKA_FCT_CUDA T * getBlockSharedExternMem() const
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
        class KernelExecutorBuilder<AccCuda, TKernel, TKernelConstrArgs...>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccCuda>::type;
            using TKernelExecutor = AccCuda::KernelExecutor<TAcceleratedKernel>;

            // Copying a kernel onto the CUDA device has some extra requirements of being trivially copyable:
            // A trivially copyable class is a class that
            // 1. Has no non-trivial copy constructors(this also requires no virtual functions or virtual bases)
            // 2. Has no non-trivial move constructors
            // 3. Has no non-trivial copy assignment operators
            // 4. Has no non-trivial move assignment operators
            // 5. Has a trivial destructor
            //
            // TODO: is_standard_layout is even stricter. Is is_trivially_copyable enough?
            static_assert(std::is_trivially_copyable<TAcceleratedKernel>::value, "The given kernel functor has to be trivially copyable to be used on a CUDA device!");

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_CPU TKernelExecutor operator()(TKernelConstrArgs && ... args) const
            {
                return TKernelExecutor(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}