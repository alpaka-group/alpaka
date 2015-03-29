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
#include <alpaka/serial/AccSerialFwd.hpp>
#include <alpaka/serial/WorkDiv.hpp>                // WorkDivSerial
#include <alpaka/serial/Idx.hpp>                    // IdxSerial
#include <alpaka/serial/Atomic.hpp>                 // AtomicSerial

// User functionality.
#include <alpaka/host/Mem.hpp>                      // Copy
#include <alpaka/serial/Stream.hpp>                 // StreamSerial
#include <alpaka/serial/Event.hpp>                  // EventSerial
#include <alpaka/serial/Device.hpp>                 // Devices

// Specialized traits.
#include <alpaka/traits/Acc.hpp>                    // AccType
#include <alpaka/traits/Exec.hpp>                   // ExecType

// Implementation details.
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>

#include <vector>                                   // std::vector
#include <cassert>                                  // assert
#include <stdexcept>                                // std::except
#include <utility>                                  // std::forward
#include <string>                                   // std::to_string
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                             // std::cout
#endif

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            // Forward declaration.
            class KernelExecSerial;

            //#############################################################################
            //! The serial accelerator.
            //!
            //! This accelerator allows serial kernel execution on the host.
            //! The block size is restricted to 1x1x1 so there is no parallelism at all.
            //#############################################################################
            class AccSerial :
                protected WorkDivSerial,
                protected IdxSerial,
                protected AtomicSerial
            {
            public:
                using MemSpace = mem::SpaceHost;
                
                friend class ::alpaka::serial::detail::KernelExecSerial;

            private:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_ACC_NO_CUDA AccSerial(
                    TWorkDiv const & workDiv) :
                        WorkDivSerial(workDiv),
                        IdxSerial(m_v3uiGridBlockIdx),
                        AtomicSerial(),
                        m_v3uiGridBlockIdx(0u)
                {}

            public:
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                // Do not copy most members because they are initialized by the executor for each execution.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccSerial(AccSerial const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccSerial(AccSerial &&) = delete;
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccSerial const &) -> AccSerial & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AccSerial() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The requested indices.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin, 
                    typename TUnit, 
                    typename TDim = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                -> DimToVecT<TDim>
                {
                    return idx::getIdx<TOrigin, TUnit, TDim>(
                        *static_cast<IdxSerial const *>(this),
                        *static_cast<WorkDivSerial const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! \return The requested extents.
                //-----------------------------------------------------------------------------
                template<
                    typename TOrigin,
                    typename TUnit,
                    typename TDim = dim::Dim3>
                ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                -> DimToVecT<TDim>
                {
                    return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                        *static_cast<WorkDivSerial const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Execute the atomic operation on the given address with the given value.
                //! \return The old value before executing the atomic operation.
                //-----------------------------------------------------------------------------
                template<
                    typename TOp,
                    typename T>
                ALPAKA_FCT_ACC auto atomicOp(
                    T * const addr,
                    T const & value) const
                -> T
                {
                    return atomic::atomicOp<TOp, T>(
                        addr,
                        value,
                        *static_cast<AtomicSerial const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all threads in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockThreads() const
                {
                    // Nothing to do in here because only one thread in a group is allowed.
                }

                //-----------------------------------------------------------------------------
                //! \return Allocates block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T, 
                    UInt TuiNumElements>
                ALPAKA_FCT_ACC_NO_CUDA auto allocBlockSharedMem() const
                -> T *
                {
                    static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                    // \TODO: C++14 std::make_unique would be better.
                    m_vvuiSharedMem.emplace_back(
                        std::unique_ptr<uint8_t[]>(
                            reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                    return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                }

                //-----------------------------------------------------------------------------
                //! \return The pointer to the externally allocated block shared memory.
                //-----------------------------------------------------------------------------
                template<
                    typename T>
                ALPAKA_FCT_ACC_NO_CUDA auto getBlockSharedExternMem() const
                -> T *
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
            //! The serial accelerator executor.
            //#############################################################################
            class KernelExecSerial :
                private AccSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FCT_HOST KernelExecSerial(
                    TWorkDiv const & workDiv, 
                    StreamSerial const &) :
                        AccSerial(workDiv)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecSerial(
                    KernelExecSerial const & other) :
                        AccSerial(static_cast<WorkDivSerial const &>(other))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecSerial(
                    KernelExecSerial && other) :
                        AccSerial(static_cast<WorkDivSerial &&>(other))
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                }
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST auto operator=(KernelExecSerial const &) -> KernelExecSerial & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                ALPAKA_FCT_HOST virtual ~KernelExecSerial() = default;
#else
                ALPAKA_FCT_HOST virtual ~KernelExecSerial() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Executes the kernel functor.
                //-----------------------------------------------------------------------------
                template<
                    typename TKernelFunctor,
                    typename... TArgs>
                ALPAKA_FCT_HOST auto operator()(
                    TKernelFunctor && kernelFunctor,
                    TArgs && ... args) const
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    Vec<3u> const v3uiGridBlockExtents(this->AccSerial::getWorkDiv<Grid, Blocks, dim::Dim3>());
                    Vec<3u> const v3uiBlockThreadExtents(this->AccSerial::getWorkDiv<Block, Threads, dim::Dim3>());

                    auto const uiBlockSharedExternMemSizeBytes(getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccSerial>(
                        v3uiBlockThreadExtents, 
                        std::forward<TArgs>(args)...));
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    std::cout << BOOST_CURRENT_FUNCTION
                        << " BlockSharedExternMemSizeBytes: " << uiBlockSharedExternMemSizeBytes << " B"
                        << std::endl;
#endif
                    this->AccSerial::m_vuiExternalSharedMem.reset(
                        new uint8_t[uiBlockSharedExternMemSizeBytes]);

                    // Execute the blocks serially.
                    for(this->AccSerial::m_v3uiGridBlockIdx[2] = 0; this->AccSerial::m_v3uiGridBlockIdx[2]<v3uiGridBlockExtents[2]; ++this->AccSerial::m_v3uiGridBlockIdx[2])
                    {
                        for(this->AccSerial::m_v3uiGridBlockIdx[1] = 0; this->AccSerial::m_v3uiGridBlockIdx[1]<v3uiGridBlockExtents[1]; ++this->AccSerial::m_v3uiGridBlockIdx[1])
                        {
                            for(this->AccSerial::m_v3uiGridBlockIdx[0] = 0; this->AccSerial::m_v3uiGridBlockIdx[0]<v3uiGridBlockExtents[0]; ++this->AccSerial::m_v3uiGridBlockIdx[0])
                            {
                                assert(v3uiBlockThreadExtents[0] == 1);
                                assert(v3uiBlockThreadExtents[1] == 1);
                                assert(v3uiBlockThreadExtents[2] == 1);

                                // There is only ever one thread in a block in the serial accelerator.
                                std::forward<TKernelFunctor>(kernelFunctor)(
                                    (*static_cast<AccSerial const *>(this)),
                                    std::forward<TArgs>(args)...);

                                // After a block has been processed, the shared memory can be deleted.
                                this->AccSerial::m_vvuiSharedMem.clear();
                            }
                        }
                    }
                    // After all blocks have been processed, the external shared memory can be deleted.
                    this->AccSerial::m_vuiExternalSharedMem.reset();
                }
            };
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The serial accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                serial::detail::KernelExecSerial>
            {
                using type = AccSerial;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The serial accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                AccSerial>
            {
                using type = serial::detail::KernelExecSerial;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The serial accelerator kernel executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                serial::detail::KernelExecSerial>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    serial::detail::KernelExecSerial const &)
                -> serial::detail::StreamSerial
                {
                    return serial::detail::StreamSerial();
                }
            };
        }
    }
}
