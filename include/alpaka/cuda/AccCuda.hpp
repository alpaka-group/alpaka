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

// Base classes.
#include <alpaka/cuda/AccCudaFwd.hpp>
#include <alpaka/cuda/WorkSize.hpp>                 // TInterfacedWorkSize
#include <alpaka/cuda/Index.hpp>                    // TInterfacedIndex
#include <alpaka/cuda/Atomic.hpp>                   // TInterfacedAtomic

// User functionality.
#include <alpaka/cuda/Memory.hpp>                   // MemCopy
#include <alpaka/cuda/Event.hpp>                    // Event
#include <alpaka/cuda/Device.hpp>                   // Devices

// Specialized templates.
#include <alpaka/interfaces/KernelExecCreator.hpp>  // KernelExecCreator

// Implementation details.
#include <alpaka/cuda/Common.hpp>
#include <alpaka/interfaces/IAcc.hpp>               // IAcc
#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>

#include <cstddef>                                  // std::size_t
#include <cstdint>                                  // std::uint32_t
#include <stdexcept>                                // std::except
#include <string>                                   // std::to_string
#include <utility>                                  // std::forward
#include <tuple>                                    // std::tuple
#ifdef ALPAKA_DEBUG
    #include <iostream>                             // std::cout
#endif

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

// Workarounds.
#include <boost/predef.h>

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
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC AccCuda & operator=(AccCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC ~AccCuda() noexcept = default;

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
                // Copying a kernel onto the CUDA device has some extra requirements of being trivially copyable:
                // A trivially copyable class is a class that
                // 1. Has no non-trivial copy constructors(this also requires no virtual functions or virtual bases)
                // 2. Has no non-trivial move constructors
                // 3. Has no non-trivial copy assignment operators
                // 4. Has no non-trivial move assignment operators
                // 5. Has a trivial destructor
                //
#if BOOST_COMP_GNUC // FIXME: Find out which version > 4.9.0 does support the std::is_trivially_copyable
                // TODO: is_standard_layout is even stricter. Is is_trivially_copyable enough?
                static_assert(std::is_trivially_copyable<TAcceleratedKernel>::value, "The given kernel functor has to be trivially copyable to be used on a CUDA device!");
#endif
                static_assert(std::is_base_of<IAcc<AccCuda>, TAcceleratedKernel>::value, "The TAcceleratedKernel for the cuda::detail::KernelExecutor has to inherit from IAcc<AccCuda>!");

            public:
                using TAcc = AccCuda;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<typename TWorkSize, typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutor(IWorkSize<TWorkSize> const & workSize, TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...)
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccCuda::KernelExecutor()" << std::endl;
#endif
                    /*auto const uiNumKernelsPerBlock(workSize.template getSize<Block, Kernels, Linear>());
                    auto const uiMaxKernelsPerBlock(AccCuda::getSizeBlockKernelsLinearMax());
                    if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                    {
                        throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the CUDA accelerator!").c_str());
                    }*/

                    m_v3uiSizeGridBlocks = workSize.template getSize<Grid, Blocks, D3>();
                    m_v3uiSizeBlockKernels = workSize.template getSize<Block, Kernels, D3>();
#ifdef ALPAKA_DEBUG
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
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutor & operator=(KernelExecutor const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~KernelExecutor() noexcept = default;

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<typename... TArgs>
                ALPAKA_FCT_HOST void operator()(TArgs && ... args) const
                {
                    dim3 gridDim(m_v3uiSizeGridBlocks[0], m_v3uiSizeGridBlocks[1], m_v3uiSizeGridBlocks[2]);
                    dim3 blockDim(m_v3uiSizeBlockKernels[0], m_v3uiSizeBlockKernels[1], m_v3uiSizeBlockKernels[2]);
#ifdef ALPAKA_DEBUG
                    //std::cout << "GridBlocks: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
                    //std::cout << "BlockKernels: (" <<  << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
#endif
                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(m_v3uiSizeBlockKernels, std::forward<TArgs>(args)...));

                    detail::cudaKernel<<<gridDim, blockDim, uiBlockSharedExternMemSizeBytes>>>(*static_cast<TAcceleratedKernel const *>(this), args...);
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccCuda::KernelExecutor::operator()" << std::endl;
#endif
                }

            private:
                vec<3u> m_v3uiSizeGridBlocks;
                vec<3u> m_v3uiSizeBlockKernels;
            };
        }
    }

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
            using KernelExecutorExtent = KernelExecutorExtent<cuda::detail::KernelExecutor<TAcceleratedKernel>, TKernelConstrArgs...>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST KernelExecutorExtent operator()(TKernelConstrArgs && ... args) const
            {
                return KernelExecutorExtent(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
