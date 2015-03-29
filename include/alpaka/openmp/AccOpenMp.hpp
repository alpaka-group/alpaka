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
#include <alpaka/openmp/AccOpenMpFwd.hpp>
#include <alpaka/openmp/WorkDiv.hpp>                // WorkDivOpenMp
#include <alpaka/openmp/Idx.hpp>                    // IdxOpenMp
#include <alpaka/openmp/Atomic.hpp>                 // AtomicOpenMp

// User functionality.
#include <alpaka/host/Mem.hpp>                      // Copy
#include <alpaka/openmp/Stream.hpp>                 // StreamOpenMp
#include <alpaka/openmp/Event.hpp>                  // EventOpenMp
#include <alpaka/openmp/Device.hpp>                 // Devices

// Specialized traits.
#include <alpaka/traits/Acc.hpp>                    // AccType
#include <alpaka/traits/Exec.hpp>                   // ExecType

// Implementation details.
#include <alpaka/openmp/Common.hpp>
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>

#include <cstdint>                                  // std::uint32_t
#include <vector>                                   // std::vector
#include <cassert>                                  // assert
#include <stdexcept>                                // std::runtime_error
#include <string>                                   // std::to_string
#include <utility>                                  // std::move, std::forward

namespace alpaka
{
    namespace openmp
    {
        namespace detail
        {
            class KernelExecOpenMp;

            //#############################################################################
            //! The OpenMP accelerator.
            //!
            //! This accelerator allows parallel kernel execution on the host.
            // \TODO: Offloading?
            //! It uses OpenMP to implement the parallelism.
            //#############################################################################
            class AccOpenMp :
                protected WorkDivOpenMp,
                protected IdxOpenMp,
                protected AtomicOpenMp
            {
            public:
                using MemSpace = mem::SpaceHost;
                
                friend class ::alpaka::openmp::detail::KernelExecOpenMp;
                
            private:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp(
                    TWorkDiv const & workDiv) :
                        WorkDivOpenMp(workDiv),
                        IdxOpenMp(m_v3uiGridBlockIdx),
                        AtomicOpenMp(),
                        m_v3uiGridBlockIdx(0u)
                {}

            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp(AccOpenMp const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp(AccOpenMp &&) = delete;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccOpenMp & operator=(AccOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AccOpenMp() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The requested indices.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin, 
                    typename TUnit, 
                    typename TDim = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA DimToVecT<TDim> getIdx() const
                {
                    return idx::getIdx<TOrigin, TUnit, TDim>(
                        *static_cast<IdxOpenMp const *>(this),
                        *static_cast<WorkDivOpenMp const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! \return The requested extents.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin,
                    typename TUnit,
                    typename TDim = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA DimToVecT<TDim> getWorkDiv() const
                {
                    return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                        *static_cast<WorkDivOpenMp const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Execute the atomic operation on the given address with the given value.
                //! \return The old value before executing the atomic operation.
                //-----------------------------------------------------------------------------
                template<
                    typename TOp,
                    typename T>
                ALPAKA_FCT_ACC T atomicOp(
                    T * const addr,
                    T const & value) const
                {
                    return atomic::atomicOp<TOp, T>(
                        addr,
                        value,
                        *static_cast<AtomicOpenMp const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all threads in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockThreads() const
                {
                    #pragma omp barrier
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T, 
                    UInt TuiNumElements>
                ALPAKA_FCT_ACC_NO_CUDA T * allocBlockSharedMem() const
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    // Assure that all threads have executed the return of the last allocBlockSharedMem function (if there was one before).
                    syncBlockThreads();

                    // Arbitrary decision: The thread with id 0 has to allocate the memory.
                    if(::omp_get_thread_num() == 0)
                    {
                        // \TODO: C++14 std::make_unique would be better.
                        m_vvuiSharedMem.emplace_back(
                            std::unique_ptr<uint8_t[]>(
                                reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                    }
                    syncBlockThreads();

                    return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T>
                ALPAKA_FCT_ACC_NO_CUDA T * getBlockSharedExternMem() const
                {
                    return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                }

#ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
            protected:
#else
            private:
#endif
                // getIdx
                Vec<3u> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                // allocBlockSharedMem
                std::vector<
                    std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                // getBlockSharedExternMem
                std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
            }; 

            //#############################################################################
            //! The OpenMP accelerator executor.
            //#############################################################################
            class KernelExecOpenMp :
                private AccOpenMp
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_HOST KernelExecOpenMp(
                    TWorkDiv const & workDiv, 
                    StreamOpenMp const &) :
                        AccOpenMp(workDiv)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecOpenMp(
                    KernelExecOpenMp const & other) :
                        AccOpenMp(static_cast<WorkDivOpenMp const &>(other))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecOpenMp(
                    KernelExecOpenMp && other) :
                        AccOpenMp(static_cast<WorkDivOpenMp &&>(other))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecOpenMp & operator=(KernelExecOpenMp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                ALPAKA_FCT_HOST virtual ~KernelExecOpenMp() = default;
#else
                ALPAKA_FCT_HOST virtual ~KernelExecOpenMp() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Executes the kernel functor.
                //-----------------------------------------------------------------------------
                template<
                    typename TKernelFunctor,
                    typename... TArgs>
                ALPAKA_FCT_HOST void operator()(
                    TKernelFunctor && kernelFunctor,
                    TArgs && ... args) const
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    Vec<3u> const v3uiGridBlockExtents(this->AccOpenMp::getWorkDiv<Grid, Blocks, dim::Dim3>());
                    Vec<3u> const v3uiBlockThreadExtents(this->AccOpenMp::getWorkDiv<Block, Threads, dim::Dim3>());

                    auto const uiBlockSharedExternMemSizeBytes(getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccOpenMp>(
                        v3uiBlockThreadExtents, 
                        std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                        << std::endl;
#endif
                    this->AccOpenMp::m_vuiExternalSharedMem.reset(
                        new uint8_t[uiBlockSharedExternMemSizeBytes]);

                    // The number of threads in this block.
                    auto const uiNumThreadsInBlock(this->AccOpenMp::getWorkDiv<Block, Threads, dim::Dim1>()[0]);

                    // Execute the blocks serially.
                    for(this->AccOpenMp::m_v3uiGridBlockIdx[2] = 0; this->AccOpenMp::m_v3uiGridBlockIdx[2]<v3uiGridBlockExtents[2]; ++this->AccOpenMp::m_v3uiGridBlockIdx[2])
                    {
                        for(this->AccOpenMp::m_v3uiGridBlockIdx[1] = 0; this->AccOpenMp::m_v3uiGridBlockIdx[1]<v3uiGridBlockExtents[1]; ++this->AccOpenMp::m_v3uiGridBlockIdx[1])
                        {
                            for(this->AccOpenMp::m_v3uiGridBlockIdx[0] = 0; this->AccOpenMp::m_v3uiGridBlockIdx[0]<v3uiGridBlockExtents[0]; ++this->AccOpenMp::m_v3uiGridBlockIdx[0])
                            {
                                // Execute the threads in parallel.

                                // Force the environment to use the given number of threads.
                                ::omp_set_dynamic(0);

                                // Parallel execution of the threads in a block is required because when syncBlockThreads is called all of them have to be done with their work up to this line.
                                // So we have to spawn one OS thread per thread in a block.
                                // 'omp for' is not useful because it is meant for cases where multiple iterations are executed by one thread but in our case a 1:1 mapping is required.
                                // Therefore we use 'omp parallel' with the specified number of threads in a block.
                                //
                                // \TODO: Does this hinder executing multiple threads in parallel because their block sizes/omp thread numbers are interfering? Is this num_threads global? Is this a real use case? 
                                #pragma omp parallel num_threads(static_cast<int>(uiNumThreadsInBlock))
                                {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                                    if((::omp_get_thread_num() == 0) && (this->AccOpenMp::m_v3uiGridBlockIdx[2] == 0) && (this->AccOpenMp::m_v3uiGridBlockIdx[1] == 0) && (this->AccOpenMp::m_v3uiGridBlockIdx[0] == 0))
                                    {
                                        assert(::omp_get_num_threads()>=0);
                                        auto const uiNumThreads(static_cast<decltype(uiNumThreadsInBlock)>(::omp_get_num_threads()));
                                        std::cout << "omp_get_num_threads: " << uiNumThreads << std::endl;
                                        if(uiNumThreads != uiNumThreadsInBlock)
                                        {
                                            throw std::runtime_error("The OpenMP runtime did not use the number of threads that had been required!");
                                        }
                                    }
#endif
                                    std::forward<TKernelFunctor>(kernelFunctor)(
                                        (*static_cast<AccOpenMp const *>(this)),
                                        std::forward<TArgs>(args)...);

                                    // Wait for all threads to finish before deleting the shared memory.
                                    this->AccOpenMp::syncBlockThreads();
                                }

                                // After a block has been processed, the shared memory can be deleted.
                                this->AccOpenMp::m_vvuiSharedMem.clear();
                            }
                        }
                    }
                    // After all blocks have been processed, the external shared memory can be deleted.
                    this->AccOpenMp::m_vuiExternalSharedMem.reset();
                }
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The OpenMP accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                openmp::detail::KernelExecOpenMp>
            {
                using type = AccOpenMp;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The OpenMP accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                AccOpenMp>
            {
                using type = openmp::detail::KernelExecOpenMp;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The OpenMP accelerator kernel executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                openmp::detail::KernelExecOpenMp>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    openmp::detail::KernelExecOpenMp const &)
                -> openmp::detail::StreamOpenMp
                {
                    return openmp::detail::StreamOpenMp();
                }
            };
        }
    }
}
