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
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
#include <alpaka/accs/serial/Idx.hpp>       // IdxSerial
#include <alpaka/accs/serial/Atomic.hpp>    // AtomicSerial

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Mem.hpp>            // SpaceType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/accs/serial/Dev.hpp>       // Devices
#include <alpaka/accs/serial/Stream.hpp>    // StreamSerial
#include <alpaka/host/mem/Space.hpp>        // SpaceHost
#include <alpaka/traits/Kernel.hpp>         // BlockSharedExternMemSizeBytes

#include <vector>                           // std::vector
#include <cassert>                          // assert
#include <stdexcept>                        // std::except
#include <utility>                          // std::forward
#include <string>                           // std::to_string
#include <memory>                           // std::unique_ptr
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>                     // std::cout
#endif

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The serial accelerator.
        //-----------------------------------------------------------------------------
        namespace serial
        {
            //-----------------------------------------------------------------------------
            //! The serial accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                // Forward declaration.
                class ExecSerial;

                //#############################################################################
                //! The serial accelerator.
                //!
                //! This accelerator allows serial kernel execution on the host.
                //! The block size is restricted to 1x1x1 so there is no parallelism at all.
                //#############################################################################
                class AccSerial :
                    protected alpaka::workdiv::BasicWorkDiv,
                    protected IdxSerial,
                    protected AtomicSerial
                {
                public:
                    using MemSpace = mem::SpaceHost;

                    friend class ::alpaka::accs::serial::detail::ExecSerial;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccSerial(
                        TWorkDiv const & workDiv) :
                            alpaka::workdiv::BasicWorkDiv(workDiv),
                            IdxSerial(m_v3uiGridBlockIdx),
                            AtomicSerial(),
                            m_v3uiGridBlockIdx(Vec3<>::zeros())
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
                    -> Vec<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit, TDim>(
                            *static_cast<IdxSerial const *>(this),
                            *static_cast<alpaka::workdiv::BasicWorkDiv const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                    -> Vec<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                            *static_cast<alpaka::workdiv::BasicWorkDiv const *>(this));
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
                    Vec3<> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                    // allocBlockSharedMem
                    std::vector<
                        std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };

                //#############################################################################
                //! The serial accelerator executor.
                //#############################################################################
                class ExecSerial :
                    private AccSerial
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_HOST ExecSerial(
                        TWorkDiv const & workDiv,
                        StreamSerial const &) :
                            AccSerial(workDiv)
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecSerial(
                        ExecSerial const & other) :
                            AccSerial(static_cast<alpaka::workdiv::BasicWorkDiv const &>(other))
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
    #if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ExecSerial(
                        ExecSerial && other) :
                            AccSerial(static_cast<alpaka::workdiv::BasicWorkDiv &&>(other))
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    }
    #endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(ExecSerial const &) -> ExecSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
    #if BOOST_COMP_INTEL
                    ALPAKA_FCT_HOST virtual ~ExecSerial() = default;
    #else
                    ALPAKA_FCT_HOST virtual ~ExecSerial() noexcept = default;
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

                        Vec3<> const v3uiGridBlockExtents(this->AccSerial::getWorkDiv<Grid, Blocks, dim::Dim3>());
                        Vec3<> const v3uiBlockThreadExtents(this->AccSerial::getWorkDiv<Block, Threads, dim::Dim3>());

                        auto const uiBlockSharedExternMemSizeBytes(kernel::getBlockSharedExternMemSizeBytes<typename std::decay<TKernelFunctor>::type, AccSerial>(
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
    }

    using AccSerial = accs::serial::detail::AccSerial;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The serial accelerator kernel executor accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::serial::detail::ExecSerial>
            {
                using type = accs::serial::detail::AccSerial;
            };

            //#############################################################################
            //! The serial accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::AccSerial;
            };

            //#############################################################################
            //! The serial accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::serial::detail::AccSerial>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccSerial";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The serial accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::EventSerial;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The serial accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::ExecSerial;
            };
        }

        namespace mem
        {
            //#############################################################################
            //! The serial accelerator memory space trait specialization.
            //#############################################################################
            template<>
            struct SpaceType<
                accs::serial::detail::AccSerial>
            {
                using type = alpaka::mem::SpaceHost;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The serial accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::StreamSerial;
            };

            //#############################################################################
            //! The serial accelerator kernel executor stream get trait specialization.
            //#############################################################################
            template<>
            struct GetStream<
                accs::serial::detail::ExecSerial>
            {
                ALPAKA_FCT_HOST static auto getStream(
                    accs::serial::detail::ExecSerial const &)
                -> accs::serial::detail::StreamSerial
                {
                    return accs::serial::detail::StreamSerial(
                        accs::serial::detail::DevManSerial::getDevByIdx(0));
                }
            };
        }
    }
}
