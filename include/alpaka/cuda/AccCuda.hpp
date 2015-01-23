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
#include <alpaka/cuda/Atomic.hpp>                   // InterfacedAtomicCuda

// User functionality.
#include <alpaka/cuda/Mem.hpp>                      // MemCopy
#include <alpaka/cuda/Event.hpp>                    // Event
#include <alpaka/cuda/Stream.hpp>                   // Stream
#include <alpaka/cuda/Device.hpp>                   // Devices

// Specialized templates.
#include <alpaka/interfaces/KernelExecCreator.hpp>  // KernelExecCreator

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
#ifdef ALPAKA_DEBUG
    #include <iostream>                             // std::cout
#endif

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

// Workarounds.
#include <boost/predef.h>

namespace alpaka
{
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
            return TAcc::getWorkDiv<TOrigin, TUnit, TDimensionality>();
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
            return TAcc::getIdx<TOrigin, TUnit, TDimensionality>();
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
            return TAcc::atomicOp<TOp, T>(addr, value);
        }

        //-----------------------------------------------------------------------------
        //! Syncs all kernels in the current block.
        //-----------------------------------------------------------------------------
        ALPAKA_FCT_ACC_CUDA_ONLY void syncBlockKernels() const
        {
            return TAcc::syncBlockKernels();
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

            return TAcc::allocBlockSharedMem<T, TuiNumElements>();
        }

        //-----------------------------------------------------------------------------
        //! \return The pointer to the externally allocated block shared memory.
        //-----------------------------------------------------------------------------
        template<
            typename T>
        ALPAKA_FCT_ACC_CUDA_ONLY T * getBlockSharedExternMem() const
        {
            return TAcc::getBlockSharedExternMem<T>();
        }
    };

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
                IAcc<AccCuda> acc,
                TAcceleratedKernel accedKernel, 
                TArgs ... args)
            {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #error "Cuda device capability >= 2.0 is required!"
#endif
                accedKernel(
                    acc,
                    std::forward<TArgs>(args)...);
            }

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
                protected InterfacedAtomicCuda
            {
            public:
                using MemSpace = alpaka::mem::MemSpaceCuda;

                template<
                    typename TAcceleratedKernel>
                friend class alpaka::cuda::detail::KernelExecutorCuda;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda() :
                    WorkDivCuda(),
                    IdxCuda(),
                    InterfacedAtomicCuda()
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_CUDA_ONLY AccCuda(AccCuda &&) = default;
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
                    return &shMem;
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T>
                ALPAKA_FCT_ACC_CUDA_ONLY T * getBlockSharedExternMem() const
                {
                    extern __shared__ uint8_t shMem[];
                    return reinterpret_cast<T*>(shMem);
                }
            };

            //#############################################################################
            //! The CUDA accelerator executor.
            //#############################################################################
            template<
                typename TAcceleratedKernel>
            class KernelExecutorCuda :
                private TAcceleratedKernel,
                private IAcc<AccCuda>
            {CUDA accelerator
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
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...)
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccCuda::KernelExecutorCuda()" << std::endl;
#endif
                    m_v3uiGridBlocksExtents = workdiv::getWorkDiv<Grid, Blocks, Dim3>(workDiv);
                    m_v3uiBlockKernelsExtents = workdiv::getWorkDiv<Block, Kernels, Dim3>(workDiv);

                    // TODO: Check that (sizeof(TAcceleratedKernel) * m_v3uiBlockKernelsExtents.prod()) < available memory size

#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccCuda::KernelExecutorCuda()" << std::endl;
#endif
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorCuda(KernelExecutorCuda const &) = default;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorCuda(KernelExecutorCuda &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorCuda & operator=(KernelExecutorCuda const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST virtual ~KernelExecutorCuda() noexcept = default;

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<
                    typename... TArgs>
                ALPAKA_FCT_HOST void operator()(
                    TArgs && ... args) const
                {
                    dim3 gridDim(m_v3uiGridBlocksExtents[0], m_v3uiGridBlocksExtents[1], m_v3uiGridBlocksExtents[2]);
                    dim3 blockDim(m_v3uiBlockKernelsExtents[0], m_v3uiBlockKernelsExtents[1], m_v3uiBlockKernelsExtents[2]);

                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(m_v3uiBlockKernelsExtents, std::forward<TArgs>(args)...));

                    ALPAKA_CUDA_CHECK(cudaConfigureCall(gridDim, blockDim, uiBlockSharedExternMemSizeBytes, m_Stream.m_cudaStream));
                    pushCudaKernelArgument<0>(
                        (*static_cast<IAcc<AccCuda> const *>(this)),
                        std::ref(*static_cast<TAcceleratedKernel const *>(this)), 
                        std::forward<TArgs>(args)...);
                    ALPAKA_CUDA_CHECK(cudaLaunch(cudaKernel<TAcceleratedKernel, TArgs...>));
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccCuda::KernelExecutorCuda::operator()" << std::endl;
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
                    ALPAKA_CUDA_CHECK(cudaSetupArgument(&arg0, sizeof(arg0), TuiOffset));

                    // Push the rest of the arguments recursively.
                    pushCudaKernelArgument<TuiOffset+sizeof(arg0)>(std::forward<TArgs>(args)...);
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
