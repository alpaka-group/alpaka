/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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
#include <alpaka/cuda/WorkExtent.hpp>               // TInterfacedWorkExtent
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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error "Cuda device capability >= 2.0 is required!"
#endif
                accedKernel(std::forward<TArgs>(args)...);
            }

            template<typename TAcceleratedKernel>
            class KernelExecutor;

            //#############################################################################
            //! The CUDA accelerator.
            //!
            //! This accelerator allows parallel kernel execution on devices supporting CUDA.
            //#############################################################################
            class AccCuda :
                protected TInterfacedWorkExtent,
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
                    TInterfacedWorkExtent(),
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
                        *static_cast<TInterfacedWorkExtent const *>(this));
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
#if (!BOOST_COMP_GNUC) || (BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(5, 0, 0))
                static_assert(std::is_trivially_copyable<TAcceleratedKernel>::value, "The given kernel functor has to fulfill is_trivially_copyable to be used on a CUDA device!");
#endif
                static_assert(std::is_base_of<IAcc<AccCuda>, TAcceleratedKernel>::value, "The TAcceleratedKernel for the cuda::detail::KernelExecutor has to inherit from IAcc<AccCuda>!");

            public:
                using TAcc = AccCuda;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<typename TWorkExtent, typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutor(IWorkExtent<TWorkExtent> const & workExtent, TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...)
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccCuda::KernelExecutor()" << std::endl;
#endif
                    /*auto const uiNumKernelsPerBlock(workExtent.template getExtent<Block, Kernels, Linear>());
                    auto const uiMaxKernelsPerBlock(AccCuda::getExtentBlockKernelsLinearMax());
                    if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                    {
                        throw std::runtime_error(("The given block kernels count '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the CUDA accelerator!").c_str());
                    }*/

                    m_v3uiGridBlocksExtent = workExtent.template getExtent<Grid, Blocks, D3>();
                    m_v3uiBlockKernelsExtent = workExtent.template getExtent<Block, Kernels, D3>();

                    // TODO: Check that (sizeof(TAcceleratedKernel) * m_v3uiBlockKernelsExtent.prod()) < available memory size

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
                    dim3 gridDim(m_v3uiGridBlocksExtent[0], m_v3uiGridBlocksExtent[1], m_v3uiGridBlocksExtent[2]);
                    dim3 blockDim(m_v3uiBlockKernelsExtent[0], m_v3uiBlockKernelsExtent[1], m_v3uiBlockKernelsExtent[2]);

                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(m_v3uiBlockKernelsExtent, std::forward<TArgs>(args)...));

                    ALPAKA_CUDA_CHECK(cudaConfigureCall(gridDim, blockDim, uiBlockSharedExternMemSizeBytes));
                    pushCudaKernelArgument<0>(std::ref(*static_cast<TAcceleratedKernel const *>(this)), std::forward<TArgs>(args)...);
                    ALPAKA_CUDA_CHECK(cudaLaunch(cudaKernel<TAcceleratedKernel, TArgs...>));
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccCuda::KernelExecutor::operator()" << std::endl;
#endif
                }

            private:
                //-----------------------------------------------------------------------------
                //! Push kernel arguments.
                //-----------------------------------------------------------------------------
                template<std::size_t TuiOffset>
                void pushCudaKernelArgument() const
                {
                    // The base case to push zero arguments.
                }
                //-----------------------------------------------------------------------------
                //! Push kernel arguments.
                //-----------------------------------------------------------------------------
                template<std::size_t TuiOffset, typename T0, typename... TArgs>
                void pushCudaKernelArgument(T0 && arg0, TArgs && ... args) const
                {
                    // Push the first argument.
                    ALPAKA_CUDA_CHECK(cudaSetupArgument(&arg0, sizeof(arg0), TuiOffset));

                    // Push the rest of the arguments recursively.
                    pushCudaKernelArgument<TuiOffset+sizeof(arg0)>(std::forward<TArgs>(args)...);
                }

            private:
                vec<3u> m_v3uiGridBlocksExtent;
                vec<3u> m_v3uiBlockKernelsExtent;
            };
        }
    }

    //#############################################################################
    //! The CUDA accelerator interface.
    //!
    //! This specialization is required because the functions are not allowed to be declared ALPAKA_FCT_HOST_ACC but are required to be ALPAKA_FCT_ACC only.
    //#############################################################################
    template<>
    class IAcc<AccCuda> :
        protected AccCuda
    {
        using TAcc = AccCuda;
    public:
        //-----------------------------------------------------------------------------
        //! \return The requested extent.
        //-----------------------------------------------------------------------------
        template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
        ALPAKA_FCT_ACC typename detail::DimToRetType<TDimensionality>::type getExtent() const
        {
            return TAcc::getExtent<TOrigin, TUnit, TDimensionality>();
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
        // \TODO: How to assure that the kernel does not hold pointers to host memory?
        //#############################################################################
        template<typename TKernel, typename... TKernelConstrArgs>
        class KernelExecCreator<AccCuda, TKernel, TKernelConstrArgs...>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccCuda>::type;
            using TKernelExecutorExtent = KernelExecutorExtent<cuda::detail::KernelExecutor<TAcceleratedKernel>, TKernelConstrArgs...>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST TKernelExecutorExtent operator()(TKernelConstrArgs && ... args) const
            {
                return TKernelExecutorExtent(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}
