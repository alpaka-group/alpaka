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

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // acc::traits::AccType
#include <alpaka/dev/Traits.hpp>                // dev::traits::DevType
#include <alpaka/dim/Traits.hpp>                // dim::traits::DimType
#include <alpaka/exec/Traits.hpp>               // exec::traits::ExecType
#include <alpaka/size/Traits.hpp>               // size::traits::SizeType

// Implementation details.
#include <alpaka/acc/AccCpuOmp2Blocks.hpp>      // acc::AccCpuOmp2Blocks
#include <alpaka/dev/DevCpu.hpp>                // dev::DevCpu
#include <alpaka/kernel/Traits.hpp>             // kernel::getBlockSharedExternMemSizeBytes
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers

#include <alpaka/core/OpenMp.hpp>
#include <alpaka/core/MapIdx.hpp>               // core::mapIdx
#include <alpaka/core/ApplyTuple.hpp>           // core::Apply

#include <boost/align.hpp>                      // boost::aligned_alloc

#include <cassert>                              // assert
#include <stdexcept>                            // std::runtime_error
#include <tuple>                                // std::tuple
#include <type_traits>                          // std::decay
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                         // std::cout
#endif

namespace alpaka
{
    namespace exec
    {
        //#############################################################################
        //! The CPU OpenMP 2.0 block accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOmp2Blocks final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuOmp2Blocks(
                TWorkDiv && workDiv,
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args) :
                    workdiv::WorkDivMembers<TDim, TSize>(std::forward<TWorkDiv>(workDiv)),
                    m_kernelFnObj(kernelFnObj),
                    m_args(args...)
            {

                static_assert(
                    dim::Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                    "The work division and the executor have to be of the same dimensionality!");
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuOmp2Blocks(ExecCpuOmp2Blocks const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuOmp2Blocks(ExecCpuOmp2Blocks &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOmp2Blocks const &) -> ExecCpuOmp2Blocks & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOmp2Blocks &&) -> ExecCpuOmp2Blocks & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ExecCpuOmp2Blocks() = default;

            //-----------------------------------------------------------------------------
            //! Executes the kernel function object.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator()() const
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const gridBlockExtents(
                    workdiv::getWorkDiv<Grid, Blocks>(*this));
                auto const blockThreadExtents(
                    workdiv::getWorkDiv<Block, Threads>(*this));

                // Get the size of the block shared extern memory.
                auto const blockSharedExternMemSizeBytes(
                    core::apply(
                        [&](TArgs const & ... args)
                        {
                            return
                                kernel::getBlockSharedExternMemSizeBytes<
                                    TKernelFnObj,
                                    acc::AccCpuOmp2Blocks<TDim, TSize>>(
                                        blockThreadExtents,
                                        args...);
                        },
                        m_args));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << BOOST_CURRENT_FUNCTION
                    << " BlockSharedExternMemSizeBytes: " << blockSharedExternMemSizeBytes << " B" << std::endl;
#endif
                // Bind all arguments except the accelerator.
                // TODO: With C++14 we could create a perfectly argument forwarding function object within the constructor.
                auto const boundKernelFnObj(
                    core::apply(
                        [this](TArgs const & ... args)
                        {
                            return
                                std::bind(
                                    std::ref(m_kernelFnObj),
                                    std::placeholders::_1,
                                    std::ref(args)...);
                        },
                        m_args));

                // The number of blocks in the grid.
                TSize const numBlocksInGrid(gridBlockExtents.prod());
                // There is only ever one thread in a block in the OpenMP 2.0 block accelerator.
                assert(blockThreadExtents.prod() == 1u);

                // Force the environment to use the given number of threads.
                int const ompIsDynamic(::omp_get_dynamic());
                ::omp_set_dynamic(0);

                // Execute the blocks in parallel.
                // NOTE: Setting num_threads(number_of_cores) instead of the default thread number does not improve performance.
                #pragma omp parallel
                {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    // The first thread does some debug logging.
                    if(::omp_get_thread_num() == 0)
                    {
                        int const numThreads(::omp_get_num_threads());
                        std::cout << BOOST_CURRENT_FUNCTION << " omp_get_num_threads: " << numThreads << std::endl;
                    }
#endif
                    acc::AccCpuOmp2Blocks<TDim, TSize> acc(*static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this));

                    if(blockSharedExternMemSizeBytes > 0u)
                    {
                        acc.m_externalSharedMem.reset(
                            reinterpret_cast<uint8_t *>(
                                boost::alignment::aligned_alloc(16u, blockSharedExternMemSizeBytes)));
                    }

                    // NOTE: schedule(static) does not improve performance.
#if _OPENMP < 200805    // For OpenMP < 3.0 you have to declare the loop index (a signed integer) outside of the loop header.
                    std::intmax_t iNumBlocksInGrid(static_cast<std::intmax_t>(numBlocksInGrid));
                    std::intmax_t i;
                    #pragma omp for nowait schedule(guided)
                    for(i = 0; i < iNumBlocksInGrid; ++i)
#else
                    #pragma omp for nowait schedule(guided)
                    for(TSize i = 0; i < numBlocksInGrid; ++i)
#endif
                    {
                        acc.m_gridBlockIdx =
                            core::mapIdx<TDim::value>(
#if _OPENMP < 200805
                                Vec1<TSize>(static_cast<TSize>(i)),
#else
                                Vec1<TSize>(i),
#endif
                                gridBlockExtents);

                        boundKernelFnObj(
                            acc);

                        // After a block has been processed, the shared memory has to be deleted.
                        block::shared::freeMem(acc);
                    }

                    // After all blocks have been processed, the external shared memory has to be deleted.
                    acc.m_externalSharedMem.reset();
                }

                // Reset the dynamic thread number setting.
                ::omp_set_dynamic(ompIsDynamic);
            }

            TKernelFnObj m_kernelFnObj;
            std::tuple<TArgs...> m_args;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOmp2Blocks<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevManType<
                exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevManCpu;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TDim;
            };
        }
    }
    namespace exec
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 grid block executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 block executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuOmp2Blocks<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}
