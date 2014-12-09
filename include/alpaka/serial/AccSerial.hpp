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
#include <alpaka/serial/AccSerialFwd.hpp>
#include <alpaka/serial/WorkSize.hpp>               // TInterfacedWorkSize
#include <alpaka/serial/Index.hpp>                  // TInterfacedIndex
#include <alpaka/serial/Atomic.hpp>                 // TInterfacedAtomic

// User functionality.
#include <alpaka/host/Memory.hpp>                   // MemCopy
#include <alpaka/serial/Event.hpp>                  // Event
#include <alpaka/serial/Device.hpp>                 // Devices

// Specialized templates.
#include <alpaka/interfaces/KernelExecCreator.hpp>  // KernelExecCreator

// Implementation details.
#include <alpaka/interfaces/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>

#include <cstddef>                                  // std::size_t
#include <vector>                                   // std::vector
#include <cassert>                                  // assert
#include <stdexcept>                                // std::except
#include <utility>                                  // std::forward
#include <string>                                   // std::to_string
#ifdef ALPAKA_DEBUG
    #include <iostream>                             // std::cout
#endif

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            // Forward declaration.
            template<typename TAcceleratedKernel>
            class KernelExecutor;

            //#############################################################################
            //! The base class for all non accelerated kernels.
            //#############################################################################
            class AccSerial :
                protected TInterfacedWorkSize,
                protected TInterfacedIndex,
                protected TInterfacedAtomic
            {
            public:
                using MemorySpace = MemorySpaceHost;

                template<typename TAcceleratedKernel>
                friend class alpaka::serial::detail::KernelExecutor;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccSerial() :
                    TInterfacedWorkSize(),
                    TInterfacedIndex(m_v3uiGridBlockIdx, m_v3uiBlockKernelIdx),
                    TInterfacedAtomic()
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccSerial(AccSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccSerial(AccSerial &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy-assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST AccSerial & operator=(AccSerial const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~AccSerial() noexcept = default;

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
                ALPAKA_FCT_HOST typename alpaka::detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return this->TInterfacedIndex::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<TInterfacedWorkSize const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST void syncBlockKernels() const
                {
                    // Nothing to do in here because only one thread in a group is allowed.
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T, std::size_t TuiNumElements>
                ALPAKA_FCT_HOST T * allocBlockSharedMem() const
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    // TODO: Optimize: do not initialize the memory on allocation like std::vector does!
                    m_vvuiSharedMem.emplace_back(TuiNumElements);
                    return reinterpret_cast<T*>(m_vvuiSharedMem.back().data());
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_HOST T * getBlockSharedExternMem() const
                {
                    return reinterpret_cast<T*>(m_vuiExternalSharedMem.data());
                }

#ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
            protected:
#else
            private:
#endif
                // getIdx
                vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.
                vec<3u> mutable m_v3uiBlockKernelIdx;                       //!< The index of the currently executed kernel.

                // allocBlockSharedMem
                std::vector<std::vector<uint8_t>> mutable m_vvuiSharedMem;  //!< Block shared memory.

                // getBlockSharedExternMem
                std::vector<uint8_t> mutable m_vuiExternalSharedMem;        //!< External block shared memory.
            };

            //#############################################################################
            //! The executor for an accelerated serial kernel.
            //#############################################################################
            template<typename TAcceleratedKernel>
            class KernelExecutor :
                private TAcceleratedKernel
            {
                static_assert(std::is_base_of<IAcc<AccSerial>, TAcceleratedKernel>::value, "The TAcceleratedKernel for the serial::detail::KernelExecutor has to inherit from IAcc<AccSerial>!");

            public:
                using TAcc = AccSerial;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<typename TWorkSize, typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutor(IWorkSize<TWorkSize> const & workSize, TKernelConstrArgs && ... args) :
                    TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...)
                {
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccSerial::KernelExecutor()" << std::endl;
#endif
                    (*const_cast<TInterfacedWorkSize*>(static_cast<TInterfacedWorkSize const *>(this))) = workSize;

                    /*auto const uiNumKernelsPerBlock(workSize.template getSize<Block, Kernels, Linear>());
                    auto const uiMaxKernelsPerBlock(AccSerial::getSizeBlockKernelsLinearMax());
                    if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                    {
                        throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the serial accelerator!").c_str());
                    }*/

                    m_v3uiSizeGridBlocks = workSize.template getSize<Grid, Blocks, D3>();
                    m_v3uiSizeBlockKernels = workSize.template getSize<Block, Kernels, D3>();
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccSerial::KernelExecutor()" << std::endl;
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
#ifdef ALPAKA_DEBUG
                    std::cout << "[+] AccSerial::KernelExecutor::operator()" << std::endl;
#endif
                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(m_v3uiSizeBlockKernels, std::forward<TArgs>(args)...));
                    this->AccSerial::m_vuiExternalSharedMem.resize(uiBlockSharedExternMemSizeBytes);
#ifdef ALPAKA_DEBUG
                    //std::cout << "GridBlocks: " << v3uiSizeGridBlocks << " BlockKernels: " << v3uiSizeBlockKernels << std::endl;
#endif
                    // Execute the blocks serially.
                    for(std::uint32_t bz(0); bz<m_v3uiSizeGridBlocks[2]; ++bz)
                    {
                        this->AccSerial::m_v3uiGridBlockIdx[2] = bz;
                        for(std::uint32_t by(0); by<m_v3uiSizeGridBlocks[1]; ++by)
                        {
                            this->AccSerial::m_v3uiGridBlockIdx[1] = by;
                            for(std::uint32_t bx(0); bx<m_v3uiSizeGridBlocks[0]; ++bx)
                            {
                                this->AccSerial::m_v3uiGridBlockIdx[0] = bx;

                                // Execute the kernels serially.
                                for(std::uint32_t tz(0); tz<m_v3uiSizeBlockKernels[2]; ++tz)
                                {
                                    this->AccSerial::m_v3uiBlockKernelIdx[2] = tz;
                                    for(std::uint32_t ty(0); ty<m_v3uiSizeBlockKernels[1]; ++ty)
                                    {
                                        this->AccSerial::m_v3uiBlockKernelIdx[1] = ty;
                                        for(std::uint32_t tx(0); tx<m_v3uiSizeBlockKernels[0]; ++tx)
                                        {
                                            this->AccSerial::m_v3uiBlockKernelIdx[0] = tx;

                                            this->TAcceleratedKernel::operator()(std::forward<TArgs>(args)...);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // After all blocks have been processed, the shared memory can be deleted.
                    this->AccSerial::m_vvuiSharedMem.clear();
                    this->AccSerial::m_vuiExternalSharedMem.clear();
#ifdef ALPAKA_DEBUG
                    std::cout << "[-] AccSerial::KernelExecutor::operator()" << std::endl;
#endif
                }

            private:
                vec<3u> m_v3uiSizeGridBlocks;
                vec<3u> m_v3uiSizeBlockKernels;
            };
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The serial kernel executor builder.
        //#############################################################################
        template<typename TKernel, typename... TKernelConstrArgs>
        class KernelExecCreator<AccSerial, TKernel, TKernelConstrArgs...>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccSerial>::type;
            using KernelExecutorExtent = KernelExecutorExtent<serial::detail::KernelExecutor<TAcceleratedKernel>, TKernelConstrArgs...>;

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
