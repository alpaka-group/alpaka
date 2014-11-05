/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of acc.
*
* acc is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* acc is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with acc.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <acc/IAcc.hpp>                        // IAcc
#include <acc/KernelExecutorBuilder.hpp>    // KernelExecutorBuilder
#include <acc/WorkSize.hpp>                    // IWorkSize, WorkSizeDefault
#include <acc/Index.hpp>                    // IIndex
#include <acc/FctCudaCpu.hpp>                // ACC_FCT_CUDA

#include <cstddef>                            // std::size_t
#include <cstdint>                            // unit8_t
#include <stdexcept>                        // std::except
#include <string>                            // std::to_string
#include <sstream>                            // std::stringstream
#ifdef _DEBUG
    #include <iostream>                        // std::cout
#endif

#include <cuda.h>

#define ACC_CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){std::cerr<<"<"<<__FILE__<<">:"<<__LINE__<<std::endl; throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));}}

#define ACC_CUDA_CHECK_MSG(cmd,msg) {cudaError_t error = cmd; if(error!=cudaSuccess){std::cerr<<"<"<<__FILE__<<">:"<<__LINE__<<msg<<std::endl; throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));}}

#define ACC_CUDA_CHECK_NO_EXCEP(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("[CUDA] Error: <%s>:%i ",__FILE__,__LINE__);}}

namespace acc
{
    namespace detail
    {
        //#############################################################################
        //! This class stores the current indices as members.
        //#############################################################################
        class IndexCuda
        {
        public:
            //-----------------------------------------------------------------------------
            //! Default-constructor.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA IndexCuda() = default;

            //-----------------------------------------------------------------------------
            //! Copy-onstructor.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA IndexCuda(IndexCuda const & other) = default;

            //-----------------------------------------------------------------------------
            //! \return The thread index of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA vec<3> getIdxTileKernel() const
            {
                return {threadIdx.x, threadIdx.y, threadIdx.z};
            }
            //-----------------------------------------------------------------------------
            //! \return The block index of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA vec<3> getIdxGridTile() const
            {
                return {blockIdx.x, blockIdx.y, blockIdx.z};
            }
        };

        //#############################################################################
        //! The description of the work being accelerated.
        //! This class gets the sizes from the CUDA environment variables. Therefore it is only available from within a kernel invocation.
        //#############################################################################
        class WorkCuda
        {
        public:
            //-----------------------------------------------------------------------------
            //! Default-constructor.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA WorkCuda() = default;

            //-----------------------------------------------------------------------------
            //! \return The grid dimensions of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA vec<3> getSizeGridTiles() const
            {
#ifdef __CUDA_ARCH__
                return{gridDim.x, gridDim.y, gridDim.z};
#else
                throw std::logic_error("WorkCuda can not be used in non-CUDA Code!");
#endif
            }
            //-----------------------------------------------------------------------------
            //! \return The block dimensions of the currently executed kernel.
            //-----------------------------------------------------------------------------
            ACC_FCT_CPU_CUDA vec<3> getSizeTileKernels() const
            {
#ifdef __CUDA_ARCH__
                return{blockDim.x, blockDim.y, blockDim.z};
#else
                throw std::logic_error("WorkCuda can not be used in non-CUDA Code!");
#endif
            }
        };

        //-----------------------------------------------------------------------------
        //! The cuda kernel entry point.
        //-----------------------------------------------------------------------------
        template<typename TAccedKernel, typename... TArgs>
        __global__ void cudaKernel(TAccedKernel accedKernel, TArgs ... args)
        {
            accedKernel(std::forward<TArgs>(args)...);
        }
    }

    //#############################################################################
    //! The base class for all CUDA accelerated kernels.
    //#############################################################################
    class AccCuda :
        protected detail::IIndex<detail::IndexCuda>,
        protected detail::IWorkSize<detail::WorkCuda>
    {
        using TIndex = detail::IIndex<detail::IndexCuda>;
        using TWorkSize = detail::IWorkSize<detail::WorkCuda>;
    public:
        //-----------------------------------------------------------------------------
        //! Constructor.
        //-----------------------------------------------------------------------------
        ACC_FCT_CUDA AccCuda() = default;

        //-----------------------------------------------------------------------------
        //! \return The maximum number of kernels in each dimension of a tile allowed.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU static vec<3> getSizeTileKernelsMax()
        {
            // TODO: CC < 2.0? Get from CUDA API.
            return {1024u, 1024u, 64u};
        }
        //-----------------------------------------------------------------------------
        //! \return The maximum number of kernels in a tile allowed.
        //-----------------------------------------------------------------------------
        ACC_FCT_CPU static std::uint32_t getSizeTileKernelsLinearMax()
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

            int iNumGpus(0); //number of gpus
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
            ACC_CUDA_CHECK(cudaGetDeviceProperties(&devProp, deviceNumber));
            // Default compute mode (Multiple threads can use cudaSetDevice() with this device)
            if(devProp.computeMode == cudaComputeModeDefault)
            {
                ACC_CUDA_CHECK(cudaSetDevice(deviceNumber));
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
            ACC_CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

#ifdef _DEBUG
            std::cout << "[-] AccCuda::setDevice()" << std::endl;
#endif
        }

    protected:
        //-----------------------------------------------------------------------------
        //! \return The requested index.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ACC_FCT_CPU_CUDA typename DimToRetType<TDimensionality>::type getIdx() const
        {
            return TIndex::getIdx<TOrigin, TUnit, TDimensionality>(*static_cast<TWorkSize * const>(this));
        }

        //-----------------------------------------------------------------------------
        //! Atomic addition of integers.
        //-----------------------------------------------------------------------------
        template<typename T>
        ACC_FCT_CUDA void atomicFetchAdd(T * sum, T summand) const
        {
            atomicAdd(sum, summand);
        }

        //-----------------------------------------------------------------------------
        //! Syncs all threads in the current block.
        //-----------------------------------------------------------------------------
        ACC_FCT_CUDA void syncTileKernels() const
        {
            __syncthreads();
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the block shared memory.
        //-----------------------------------------------------------------------------
        template<typename T>
        ACC_FCT_CUDA T * getTileSharedExternMem() const
        {
            extern __shared__ uint8_t shMem[];
            //syncTileKernels();
            return reinterpret_cast<T*>(shMem);
        }

        //-----------------------------------------------------------------------------
        //! \return A pointer to the block shared memory.
        //-----------------------------------------------------------------------------
        // TODO: implement
        /*template<typename T>
        ACC_FCT_CUDA __forceinline__ T * getBlockSharedVar() const
        {
            __shared__ T shMem;
            syncTileKernels();
            return shMem;
        }*/

    public:
        //#############################################################################
        //! The executor for an accelerated serial kernel.
        //#############################################################################
        template<typename TAccedKernel>
        class KernelExecutor :
            protected TAccedKernel,
            public detail::IWorkSize<detail::WorkSizeDefault>
        {
            using TWorkSize = detail::IWorkSize<detail::WorkSizeDefault>;
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            KernelExecutor(TWorkSize workSize) :
                detail::IWorkSize<detail::WorkSizeDefault>(workSize)
            {
#ifdef _DEBUG
                std::cout << "AccCuda::KernelExecutor()" << std::endl;
#endif
            }

            //-----------------------------------------------------------------------------
            //! Executes the accelerated kernel.
            //-----------------------------------------------------------------------------
            template<typename... TArgs>
            void operator()(TArgs && ... args) const
            {
#ifdef _DEBUG
                std::cout << "[+] AccCuda::KernelExecutor::operator()" << std::endl;
#endif

                auto const uiNumKernelsPerTile(TWorkSize::getSize<Tile, Kernels, Linear>());
                auto const uiMaxKernelsPerTile(AccCuda::getSizeTileKernelsLinearMax());
                if(uiNumKernelsPerTile > uiMaxKernelsPerTile)
                {
                    throw std::runtime_error(("The given tileSize '" + std::to_string(uiNumKernelsPerTile) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerTile) + "' by the CUDA accelerator!").c_str());
                }

                auto const v3uiSizeGridTiles(TWorkSize::getSize<Grid, Tiles, D3>());
                auto const v3uiSizeTileKernels(TWorkSize::getSize<Tile, Kernels, D3>());
#ifdef _DEBUG
                //std::cout << "GridTiles: " << v3uiSizeGridTiles << " TileKernels: " << v3uiSizeTileKernels<< std::endl;
#endif
                dim3 gridDim(v3uiSizeGridTiles[0], v3uiSizeGridTiles[1], v3uiSizeGridTiles[2]);
                dim3 blockDim(v3uiSizeTileKernels[0], v3uiSizeTileKernels[1], v3uiSizeTileKernels[2]);
#ifdef _DEBUG
                //std::cout << "GridTiles: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
                //std::cout << "TileKernels: (" <<  << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
#endif
                TAccedKernel accedKernel (*this);
                detail::cudaKernel<<<gridDim, blockDim, TAccedKernel::getBlockSharedMemSizeBytes(v3uiSizeTileKernels)>>>(accedKernel, args...);
#ifdef _DEBUG
                std::cout << "[-] AccCuda::KernelExecutor::operator()" << std::endl;
#endif
            }
        };
    };

    namespace detail
    {
        //#############################################################################
        //! The specialization of the accelerator interface for CUDA.
        //#############################################################################
        template<>
        class IAcc<AccCuda> :
            public AccCuda
        {
            using TAcc = AccCuda;
        public:
            //-----------------------------------------------------------------------------
            //! \return The maximum number of kernels in a tile allowed by the underlying accelerator.
            //-----------------------------------------------------------------------------
            static inline std::uint32_t getSizeTileKernelsLinearMax()
            {
                return TAcc::getSizeTileKernelsLinearMax();
            }

            //-----------------------------------------------------------------------------
            //! \return The requested size.
            //-----------------------------------------------------------------------------
            template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
            ACC_FCT_CUDA typename DimToRetType<TDimensionality>::type getSize() const
            {
                return TAcc::getSize<TOrigin, TUnit, TDimensionality>();
            }

        protected:
            //-----------------------------------------------------------------------------
            //! \return The requested index.
            //-----------------------------------------------------------------------------
            template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
            ACC_FCT_CUDA typename DimToRetType<TDimensionality>::type getIdx() const
            {
                return TAcc::getIdx<TOrigin, TUnit, TDimensionality>();
            }

            //-----------------------------------------------------------------------------
            //! Atomic addition.
            //-----------------------------------------------------------------------------
            template<typename T>
            ACC_FCT_CUDA void atomicFetchAdd(T * sum, T summand) const
            {
                return TAcc::atomicFetchAdd<T>(sum, summand);
            }

            //-----------------------------------------------------------------------------
            //! Syncs all threads in the current block.
            //-----------------------------------------------------------------------------
            ACC_FCT_CUDA void syncTileKernels() const
            {
                return TAcc::syncTileKernels();
            }

            //-----------------------------------------------------------------------------
            //! \return The pointer to the block shared memory.
            //-----------------------------------------------------------------------------
            template<typename T>
            ACC_FCT_CUDA T * getTileSharedExternMem() const
            {
                return TAcc::getTileSharedExternMem<T>();
            }
        };

        //#############################################################################
        //! The serial kernel executor builder.
        //#############################################################################
        template<template<typename> class TKernel, typename TWorkSize>
        class KernelExecutorBuilder<AccCuda, TKernel, TWorkSize>
        {
        public:
            using TAccedKernel = TKernel<IAcc<AccCuda>>;
            using TKernelExecutor = AccCuda::KernelExecutor<TAccedKernel>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            TKernelExecutor operator()(TWorkSize workSize) const
            {
                return TKernelExecutor(workSize);
            }
        };
    }
}