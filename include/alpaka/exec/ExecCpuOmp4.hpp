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
#include <alpaka/acc/AccCpuOmp4.hpp>            // acc:AccCpuOmp4
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
        //! The CPU OpenMP 4.0 accelerator executor.
        //#############################################################################
        template<
            typename TDim,
            typename TSize,
            typename TKernelFnObj,
            typename... TArgs>
        class ExecCpuOmp4 final :
            public workdiv::WorkDivMembers<TDim, TSize>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            template<
                typename TWorkDiv>
            ALPAKA_FN_HOST ExecCpuOmp4(
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
            ALPAKA_FN_HOST ExecCpuOmp4(ExecCpuOmp4 const & other) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ExecCpuOmp4(ExecCpuOmp4 && other) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOmp4 const &) -> ExecCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(ExecCpuOmp4 &&) -> ExecCpuOmp4 & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~ExecCpuOmp4() = default;

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
                                    acc::AccCpuOmp4<TDim, TSize>>(
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
                // The number of threads in a block.
                TSize const numThreadsInBlock(blockThreadExtents.prod());

                // `When an if(scalar-expression) evaluates to false, the structured block is executed on the host.`
                #pragma omp target if(0)
                {
                    #pragma omp teams/* num_teams(numBlocksInGrid) thread_limit(numThreadsInBlock)*/
                    {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        // The first team does some checks ...
                        if((::omp_get_team_num() == 0))
                        {
                            int const iNumTeams(::omp_get_num_teams());
                            // NOTE: No std::cout in omp target!
                            printf("%s omp_get_num_teams: %d\n", BOOST_CURRENT_FUNCTION, iNumTeams);
                            if(iNumTeams <= 0)    // NOTE: No throw inside target region
                            {
                                throw std::runtime_error("The CPU OpenMP4 runtime did not use a valid number of teams!");
                            }
                        }
#endif
                        acc::AccCpuOmp4<TDim, TSize> acc(*static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this));

                        if(blockSharedExternMemSizeBytes > 0u)
                        {
                            acc.m_externalSharedMem.reset(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(16u, blockSharedExternMemSizeBytes)));
                        }

                        #pragma omp distribute
                        for(TSize b = 0u; b<numBlocksInGrid; ++b)
                        {
                            Vec1<TSize> const gridBlockIdx(b);
                            // When this is not repeated here:
                            // error: ‘gridBlockExtents’ referenced in target region does not have a mappable type
                            auto const gridBlockExtents2(
                                workdiv::getWorkDiv<Grid, Blocks>(*static_cast<workdiv::WorkDivMembers<TDim, TSize> const *>(this)));
                            acc.m_gridBlockIdx = core::mapIdx<TDim::value>(
                                gridBlockIdx,
                                gridBlockExtents2);

                            // Execute the threads in parallel.

                            // Force the environment to use the given number of threads.
                            int const ompIsDynamic(::omp_get_dynamic());
                            ::omp_set_dynamic(0);

                            // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                            // So we have to spawn one OS thread per thread in a block.
                            // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                            // Therefore we use 'omp parallel' with the specified number of threads in a block.
                            #pragma omp parallel num_threads(numThreadsInBlock)
                            {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                // The first thread does some checks in the first block executed.
                                if((::omp_get_thread_num() == 0) && (b == 0))
                                {
                                    int const numThreads(::omp_get_num_threads());
                                    // NOTE: No std::cout in omp target!
                                    printf("%s omp_get_num_threads: %d\n", BOOST_CURRENT_FUNCTION, numThreads);
                                    if(numThreads != numThreadsInBlock)
                                    {
                                        throw std::runtime_error("The CPU OpenMP4 runtime did not use the number of threads that had been required!");
                                    }
                                }
#endif
                                boundKernelFnObj(
                                    acc);

                                // Wait for all threads to finish before deleting the shared memory.
                                block::sync::syncBlockThreads(acc);
                            }

                            // Reset the dynamic thread number setting.
                            ::omp_set_dynamic(ompIsDynamic);

                            // After a block has been processed, the shared memory has to be deleted.
                            block::shared::freeMem(acc);
                        }
                        // After all blocks have been processed, the external shared memory has to be deleted.
                        acc.m_externalSharedMem.reset();
                    }
                }
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
            //! The CPU OpenMP4 executor accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct AccType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = acc::AccCpuOmp4<TDim, TSize>;
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP4 executor device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP4 executor device manager type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DevManType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP4 executor dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct DimType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
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
            //! The CPU OpenMP4 executor executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct ExecType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>,
                TKernelFnObj,
                TArgs...>
            {
                using type = exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 executor size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize,
                typename TKernelFnObj,
                typename... TArgs>
            struct SizeType<
                exec::ExecCpuOmp4<TDim, TSize, TKernelFnObj, TArgs...>>
            {
                using type = TSize;
            };
        }
    }
}
