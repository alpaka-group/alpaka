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

#include <cstddef>                          // std::size_t
#include <vector>                           // std::vector
#include <cassert>                          // assert
#include <stdexcept>                        // std::except
#include <utility>                          // std::forward
#include <string>                           // std::to_string
#ifdef _DEBUG
    #include <iostream>                     // std::cout
#endif

#include <boost/mpl/apply.hpp>              // boost::mpl::apply

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            using TInterfacedWorkSize = alpaka::IWorkSize<alpaka::detail::WorkSizeDefault>;

            //#############################################################################
            //! This class that holds the implementation details for the indexing of the serial accelerator.
            //#############################################################################
            class IndexSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU IndexSerial(
                    vec<3u> const & v3uiGridBlockIdx,
                    vec<3u> const & v3uiBlockKernelIdx) :
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx),
                    m_v3uiBlockKernelIdx(v3uiBlockKernelIdx)
                {}
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU IndexSerial(IndexSerial const & other) = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU vec<3u> getIdxBlockKernel() const
                {
                    return m_v3uiBlockKernelIdx;
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU vec<3u> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                vec<3u> const & m_v3uiGridBlockIdx;
                vec<3u> const & m_v3uiBlockKernelIdx;
            };
            using TInterfacedIndex = alpaka::detail::IIndex<IndexSerial>;

            //#############################################################################
            //! This class that holds the implementation details for the atomic operations of the serial accelerator.
            //#############################################################################
            class AtomicSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU AtomicSerial() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU AtomicSerial(AtomicSerial const & other) = default;
            };
            using TInterfacedAtomic = alpaka::detail::IAtomic<AtomicSerial>;
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The specialization to execute the requested atomic operation of the serial accelerator.
        //#############################################################################
        template<typename TOp, typename T>
        struct AtomicOp<serial::detail::AtomicSerial, TOp, T>
        {
            ALPAKA_FCT_CPU static T atomicOp(serial::detail::AtomicSerial const &, T * const addr, T const & value)
            {
                return TOp::op(addr, value);
            }
        };
    }

    namespace serial
    {
        namespace detail
        {
            //#############################################################################
            //! The base class for all non accelerated kernels.
            //#############################################################################
            class AccSerial :
                protected TInterfacedWorkSize,
                protected TInterfacedIndex,
                protected TInterfacedAtomic
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU AccSerial() :
                    TInterfacedWorkSize(),
                    TInterfacedIndex(m_v3uiGridBlockIdx, m_v3uiBlockKernelIdx),
                    TInterfacedAtomic()
                {
                }

                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in each dimension of a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU static vec<3u> getSizeBlockKernelsMax()
                {
                    auto const uiSizeBlockKernelsLinearMax(getSizeBlockKernelsLinearMax());
                    return{uiSizeBlockKernelsLinearMax, uiSizeBlockKernelsLinearMax, uiSizeBlockKernelsLinearMax};
                }
                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU static std::uint32_t getSizeBlockKernelsLinearMax()
                {
                    return 1;
                }

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
                ALPAKA_FCT_CPU typename alpaka::detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return this->TInterfacedIndex::getIdx<TOrigin, TUnit, TDimensionality>(
                        *static_cast<TInterfacedWorkSize const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU void syncBlockKernels() const
                {
                    // Nothing to do in here because only one thread in a group is allowed.
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T, std::size_t UiNumElements>
                ALPAKA_FCT_CPU T * allocBlockSharedMem() const
                {
                    static_assert(UiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    m_vvuiSharedMem.emplace_back(UiNumElements);
                    return reinterpret_cast<T*>(m_vvuiSharedMem.back().data());
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_CPU T * getBlockSharedExternMem() const
                {
                    return reinterpret_cast<T*>(m_vuiExternalSharedMem.data());
                }

            private:
                // getIdx
                vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.
                vec<3u> mutable m_v3uiBlockKernelIdx;                       //!< The index of the currently executed kernel.

                // allocBlockSharedMem
                std::vector<std::vector<uint8_t>> mutable m_vvuiSharedMem;  //!< Block shared memory.

                // getBlockSharedExternMem
                std::vector<uint8_t> mutable m_vuiExternalSharedMem;        //!< External block shared memory.

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
                    KernelExecutor(TKernelConstrArgs && ... args) :
                        TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...)
                    {
#ifdef _DEBUG
                        std::cout << "[+] AccSerial::KernelExecutor()" << std::endl;
#endif
#ifdef _DEBUG
                        std::cout << "[-] AccSerial::KernelExecutor()" << std::endl;
#endif
                    }

                    //-----------------------------------------------------------------------------
                    //! Executes the accelerated kernel.
                    //-----------------------------------------------------------------------------
                    template<typename TWorkSize, typename... TArgs>
                    void operator()(IWorkSize<TWorkSize> const & workSize, TArgs && ... args) const
                    {
#ifdef _DEBUG
                        std::cout << "[+] AccSerial::KernelExecutor::operator()" << std::endl;
#endif
                        (*const_cast<TInterfacedWorkSize*>(static_cast<TInterfacedWorkSize const *>(this))) = workSize;
                                               
                        auto const uiNumKernelsPerBlock(this->TAcceleratedKernel::template getSize<Block, Kernels, Linear>());
                        auto const uiMaxKernelsPerBlock(this->TAcceleratedKernel::getSizeBlockKernelsLinearMax());
                        if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                        {
                            throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the serial accelerator!").c_str());
                        }

                        auto const v3uiSizeBlockKernels(this->TAcceleratedKernel::template getSize<Block, Kernels, D3>());
                        this->AccSerial::m_vuiExternalSharedMem.resize(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(v3uiSizeBlockKernels));

                        auto const v3uiSizeGridBlocks(this->TAcceleratedKernel::template getSize<Grid, Blocks, D3>());
#ifdef _DEBUG
                        //std::cout << "GridBlocks: " << v3uiSizeGridBlocks << " BlockKernels: " << v3uiSizeBlockKernels << std::endl;
#endif
                        for(std::uint32_t bz(0); bz<v3uiSizeGridBlocks[2]; ++bz)
                        {
                            this->AccSerial::m_v3uiGridBlockIdx[2] = bz;
                            for(std::uint32_t by(0); by<v3uiSizeGridBlocks[1]; ++by)
                            {
                                this->AccSerial::m_v3uiGridBlockIdx[1] = by;
                                for(std::uint32_t bx(0); bx<v3uiSizeGridBlocks[0]; ++bx)
                                {
                                    this->AccSerial::m_v3uiGridBlockIdx[0] = bx;

                                    for(std::uint32_t tz(0); tz<v3uiSizeBlockKernels[2]; ++tz)
                                    {
                                        this->AccSerial::m_v3uiBlockKernelIdx[2] = tz;
                                        for(std::uint32_t ty(0); ty<v3uiSizeBlockKernels[1]; ++ty)
                                        {
                                            this->AccSerial::m_v3uiBlockKernelIdx[1] = ty;
                                            for(std::uint32_t tx(0); tx<v3uiSizeBlockKernels[0]; ++tx)
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
#ifdef _DEBUG
                        std::cout << "[-] AccSerial::KernelExecutor::operator()" << std::endl;
#endif
                    }
                };
            };
        }
    }

    using AccSerial = serial::detail::AccSerial;

    namespace detail
    {
        //#############################################################################
        //! The serial kernel executor builder.
        //#############################################################################
        template<typename TKernel, typename... TKernelConstrArgs>
        class KernelExecutorBuilder<AccSerial, TKernel, TKernelConstrArgs...>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccSerial>::type;
            using TKernelExecutor = AccSerial::KernelExecutor<TAcceleratedKernel>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            TKernelExecutor operator()(TKernelConstrArgs && ... args) const
            {
                return TKernelExecutor(std::forward<TKernelConstrArgs>(args)...);
            }
        };
    }
}