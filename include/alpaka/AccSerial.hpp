/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of of either the GNU General Public License or
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
            //#############################################################################
            //! This class stores the current indices as members.
            //#############################################################################
            class IndexSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Default-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU_CUDA IndexSerial() = default;

                //-----------------------------------------------------------------------------
                //! Copy-onstructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU_CUDA IndexSerial(IndexSerial const & other) = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU_CUDA vec<3> getIdxBlockKernel() const
                {
                    return m_v3uiBlockKernelIdx;
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU_CUDA vec<3> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                vec<3> m_v3uiBlockKernelIdx;
                vec<3> m_v3uiGridBlockIdx;
            };

            using TPackedIndex = alpaka::detail::IIndex<IndexSerial>;
            using TPackedWorkSize = alpaka::detail::IWorkSize<alpaka::detail::WorkSizeDefault>;

            //#############################################################################
            //! The base class for all non accelerated kernels.
            //#############################################################################
            class AccSerial :
                protected TPackedIndex,
                public TPackedWorkSize
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU AccSerial() :
                    TPackedIndex(),
                    TPackedWorkSize()
                {
                }

                //-----------------------------------------------------------------------------
                //! \return The maximum number of kernels in each dimension of a block allowed.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU static vec<3> getSizeBlockKernelsMax()
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
                ALPAKA_FCT_CPU_CUDA typename detail::DimToRetType<TDimensionality>::type getIdx() const
                {
                    return TPackedIndex::getIdx<TPackedWorkSize, TOrigin, TUnit, TDimensionality>(
                        *static_cast<TPackedWorkSize const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Atomic addition.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_CPU void atomicFetchAdd(T * sum, T summand) const
                {
                    auto & rsum = *sum;
                    rsum += summand;
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU void syncBlockKernels() const
                {
                    // Nothing to do in here because only one thread in a group is allowed.
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the block shared memory.
                //-----------------------------------------------------------------------------
                template<typename T>
                ALPAKA_FCT_CPU T * getBlockSharedExternMem() const
                {
                    return reinterpret_cast<T*>(m_vuiSharedMem.data());
                }

            private:
                mutable std::vector<uint8_t> m_vuiSharedMem;    //!< Block shared memory.

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
					template<typename TWorkSize2, typename TNonAcceleratedKernel>
					KernelExecutor(TWorkSize2 const & workSize, TNonAcceleratedKernel const & kernel)
                    {
                        (*static_cast<typename AccSerial::TPackedWorkSize*>(this)) = workSize;
#ifdef _DEBUG
                        std::cout << "AccSerial::KernelExecutor()" << std::endl;
#endif
                    }

                    //-----------------------------------------------------------------------------
                    //! Executes the accelerated kernel.
                    //-----------------------------------------------------------------------------
                    template<typename... TArgs>
                    void operator()(TArgs && ... args) const
                    {
#ifdef _DEBUG
                        std::cout << "[+] AccSerial::KernelExecutor::operator()" << std::endl;
#endif
                        auto const uiNumKernelsPerBlock(this->AccSerial::template getSize<Block, Kernels, Linear>());
                        auto const uiMaxKernelsPerBlock(AccSerial::getSizeBlockKernelsLinearMax());
                        if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                        {
                            throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the serial accelerator!").c_str());
                        }

                        auto const v3uiSizeGridBlocks(this->AccSerial::template getSize<Grid, Blocks, D3>());
                        auto const v3uiSizeBlockKernels(this->AccSerial::template getSize<Block, Kernels, D3>());
#ifdef _DEBUG
                        //std::cout << "GridBlocks: " << v3uiSizeGridBlocks << " BlockKernels: " << v3uiSizeBlockKernels << std::endl;
#endif

                        this->AccSerial::m_vuiSharedMem.resize(TAcceleratedKernel::getBlockSharedMemSizeBytes(v3uiSizeBlockKernels));

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
                        this->AccSerial::m_vuiSharedMem.clear();
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
        template<typename TKernel, typename TPackedWorkSize>
        class KernelExecutorBuilder<AccSerial, TKernel, TPackedWorkSize>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccSerial>::type;
            using TKernelExecutor = AccSerial::KernelExecutor<TAcceleratedKernel>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            TKernelExecutor operator()(TPackedWorkSize const & workSize, TKernel const & kernel) const
            {
				return TKernelExecutor(workSize, kernel);
            }
        };
    }
}