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

#include <cstddef>                          // std::size_t
#include <cstdint>                          // unit8_t
#include <vector>                           // std::vector
#include <cassert>                          // assert
#include <stdexcept>                        // std::except
#include <string>                           // std::to_string
#ifdef _DEBUG
    #include <iostream>                     // std::cout
#endif

#include <boost/mpl/apply.hpp>              // boost::mpl::apply

#include <omp.h>

namespace alpaka
{
    namespace openmp
    {
        namespace detail
        {
            using TPackedWorkSize = alpaka::detail::IWorkSize<alpaka::detail::WorkSizeDefault>;

            //#############################################################################
            //! This class stores the current indices as members.
            //#############################################################################
            class IndexOpenMp
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU_CUDA IndexOpenMp(
                    TPackedWorkSize const & workSize,
                    vec<3> & v3uiGridBlockIdx) :
                    m_WorkSize(workSize),
                    m_v3uiGridBlockIdx(v3uiGridBlockIdx)
                {}

                //-----------------------------------------------------------------------------
                //! Copy-onstructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU_CUDA IndexOpenMp(IndexOpenMp const & other) = default;

                //-----------------------------------------------------------------------------
                //! \return The index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU vec<3> getIdxBlockKernel() const
                {
                    vec<3> v3uiIdxBlockKernel;

                    auto const v3uiSizeBlockKernels(m_WorkSize.getSize<Block, Kernels, D3>());
                    auto const t(::omp_get_thread_num());
                    v3uiIdxBlockKernel[0] = (t % (v3uiSizeBlockKernels[1] * v3uiSizeBlockKernels[0])) % v3uiSizeBlockKernels[0];
                    v3uiIdxBlockKernel[1] = (t % (v3uiSizeBlockKernels[1] * v3uiSizeBlockKernels[0])) / v3uiSizeBlockKernels[0];
                    v3uiIdxBlockKernel[2] = (t / (v3uiSizeBlockKernels[1] * v3uiSizeBlockKernels[0]));

                    return v3uiIdxBlockKernel;
                }
                //-----------------------------------------------------------------------------
                //! \return The block index of the currently executed kernel.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU vec<3> getIdxGridBlock() const
                {
                    return m_v3uiGridBlockIdx;
                }

            private:
                TPackedWorkSize const & m_WorkSize;		//!< The mapping of thread id's to thread indices.
                vec<3> const & m_v3uiGridBlockIdx;		//!< The index of the currently executed block.
            };

            using TPackedIndex = alpaka::detail::IIndex<IndexOpenMp>;

            //#############################################################################
            //! The base class for all OpenMP accelerated kernels.
            //#############################################################################
            class AccOpenMp :
                protected TPackedIndex,
                public TPackedWorkSize
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU AccOpenMp() :
                    TPackedIndex(*static_cast<TPackedWorkSize const *>(this), m_v3uiGridBlockIdx),
                    TPackedWorkSize()
                {}

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
                    // HACK: ::omp_get_max_threads() does not return the real limit of the underlying OpenMP runtime:
                    // 'The omp_get_max_threads routine returns the value of the nthreads-var internal control variable, which is used to determine the number of threads that would form the new team, 
                    // if an active parallel region without a num_threads clause were to be encountered at that point in the program.'
                    // How to do this correctly? Is there even a way to get the hard limit apart from omp_set_num_threads(high_value) -> omp_get_max_threads()?
                    ::omp_set_num_threads(1024);
                    return ::omp_get_max_threads();
                }

            protected:
                //-----------------------------------------------------------------------------
                //! \return The requested index.
                //-----------------------------------------------------------------------------
                template<typename TOrigin, typename TUnit, typename TDimensionality = dim::D3>
                ALPAKA_FCT_CPU_CUDA typename alpaka::detail::DimToRetType<TDimensionality>::type getIdx() const
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
                    auto & rsum(*sum);
                    // NOTE: Braces or calling other functions directly after 'atomic' are not allowed!
#pragma omp atomic
                    rsum += summand;
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_CPU void syncBlockKernels() const
                {
#pragma omp barrier
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

                mutable vec<3> m_v3uiGridBlockIdx;               //!< The index of the currently executed block.

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
						(*static_cast<TPackedWorkSize*>(this)) = workSize;
						// TODO: implement!
						//(*static_cast<typename TAcceleratedKernel*>(this)) = kernel;
#ifdef _DEBUG
                        std::cout << "AccOpenMp::KernelExecutor()" << std::endl;
#endif
                    }

                    //-----------------------------------------------------------------------------
                    //! Executes the accelerated kernel.
                    //-----------------------------------------------------------------------------
                    template<typename... TArgs>
                    void operator()(TArgs && ... args) const
                    {
#ifdef _DEBUG
                        std::cout << "[+] AccOpenMp::KernelExecutor::operator()" << std::endl;
#endif

                        auto const uiNumKernelsPerBlock(this->AccOpenMp::template getSize<Block, Kernels, Linear>());
                        auto const uiMaxKernelsPerBlock(AccOpenMp::getSizeBlockKernelsLinearMax());
                        if(uiNumKernelsPerBlock > uiMaxKernelsPerBlock)
                        {
                            throw std::runtime_error(("The given blockSize '" + std::to_string(uiNumKernelsPerBlock) + "' is larger then the supported maximum of '" + std::to_string(uiMaxKernelsPerBlock) + "' by the OpenMp accelerator!").c_str());
                        }

                        auto const v3uiSizeGridBlocks(this->AccOpenMp::template getSize<Grid, Blocks, D3>());
                        auto const v3uiSizeBlockKernels(this->AccOpenMp::template getSize<Block, Kernels, D3>());
#ifdef _DEBUG
                        //std::cout << "GridBlocks: " << v3uiSizeGridBlocks << " BlockKernels: " << v3uiSizeBlockKernels << std::endl;
#endif

                        this->AccOpenMp::m_vuiSharedMem.resize(TAcceleratedKernel::getBlockSharedMemSizeBytes(v3uiSizeBlockKernels));

                        // CUDA programming guide: "Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
                        // This independence requirement allows thread blocks to be scheduled in any order across any number of cores"
                        // -> We can execute them serially.
                        for(std::uint32_t bz(0); bz<v3uiSizeGridBlocks[2]; ++bz)
                        {
                            this->AccOpenMp::m_v3uiGridBlockIdx[2] = bz;
                            for(std::uint32_t by(0); by<v3uiSizeGridBlocks[1]; ++by)
                            {
                                this->AccOpenMp::m_v3uiGridBlockIdx[1] = by;
                                for(std::uint32_t bx(0); bx<v3uiSizeGridBlocks[0]; ++bx)
                                {
                                    this->AccOpenMp::m_v3uiGridBlockIdx[0] = bx;

                                    // The number of threads in this block.
                                    std::uint32_t const uiNumKernelsInBlock(this->TAcceleratedKernel::template getSize<Block, Kernels, Linear>());

                                    // Force the environment to use the given number of threads.
                                    ::omp_set_dynamic(0);

                                    // Parallelizing the threads is required because when syncBlockKernels is called all of them have to be done with their work up to this line.
                                    // So we have to spawn one real thread per thread in a block.
                                    // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our cas a 1:1 mapping is required.
                                    // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                    // FIXME: Does this hinder executing multiple kernels in parallel because their block sizes/omp thread numbers are interfering? Is this a real use case? 
#pragma omp parallel num_threads(uiNumKernelsInBlock)
                                    {
#ifdef _DEBUG
                                        if((::omp_get_thread_num() == 0) && (bz == 0) && (by == 0) && (bx == 0))
                                        {
                                            std::cout << "omp_get_num_threads: " << ::omp_get_num_threads() << std::endl;
                                        }
#endif
                                        this->TAcceleratedKernel::operator()(std::forward<TArgs>(args)...);

                                        // Wait for all threads to finish before deleting the shared memory.
                                        this->AccOpenMp::syncBlockKernels();
                                    }
                                }
                            }
                        }

                        // After all blocks have been processed, the shared memory can be deleted.
                        this->AccOpenMp::m_vuiSharedMem.clear();
#ifdef _DEBUG
                        std::cout << "[-] AccOpenMp::KernelExecutor::operator()" << std::endl;
#endif
                    }
                };
            };
        }
    }

    using AccOpenMp = openmp::detail::AccOpenMp;

    namespace detail
    {
        //#############################################################################
        //! The serial kernel executor builder.
        //#############################################################################
        template<typename TKernel, typename TWorkSize>
        class KernelExecutorBuilder<AccOpenMp, TKernel, TWorkSize>
        {
        public:
            using TAcceleratedKernel = typename boost::mpl::apply<TKernel, AccOpenMp>::type;
            using TKernelExecutor = AccOpenMp::KernelExecutor<TAcceleratedKernel>;

            //-----------------------------------------------------------------------------
            //! Creates an kernel executor for the serial accelerator.
            //-----------------------------------------------------------------------------
            TKernelExecutor operator()(TWorkSize const & workSize, TKernel const & kernel) const
            {
				return TKernelExecutor(workSize, kernel);
            }
        };
    }
}