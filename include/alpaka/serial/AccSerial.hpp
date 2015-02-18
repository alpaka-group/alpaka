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
#include <alpaka/host/Mem.hpp>                      // MemCopy
#include <alpaka/serial/Stream.hpp>                 // StreamSerial
#include <alpaka/serial/Event.hpp>                  // EventSerial
#include <alpaka/serial/Device.hpp>                 // Devices

// Specialized traits.
#include <alpaka/core/KernelExecCreator.hpp>        // KernelExecCreator

// Implementation details.
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/interfaces/IAcc.hpp>

#include <boost/mpl/apply.hpp>                      // boost::mpl::apply

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
            template<
                typename TAcceleratedKernel>
            class KernelExecutorSerial;

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
                using MemSpace = mem::MemSpaceHost;

                template<
                    typename TAcceleratedKernel>
                friend class KernelExecutorSerial;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA AccSerial() :
                    WorkDivSerial(),
                    IdxSerial(m_v3uiGridBlockIdx),
                    AtomicSerial()
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                // Do not copy most members because they are initialized by the executor for each accelerated execution.
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
                ALPAKA_FCT_ACC_NO_CUDA AccSerial & operator=(AccSerial const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA virtual ~AccSerial() noexcept = default;

            protected:
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
                ALPAKA_FCT_ACC_NO_CUDA DimToVecT<TDim> getWorkDiv() const
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
                ALPAKA_FCT_ACC T atomicOp(
                    T * const addr,
                    T const & value) const
                {
                    return atomic::atomicOp<TOp, T>(
                        addr,
                        value,
                        *static_cast<AtomicSerial const *>(this));
                }

                //-----------------------------------------------------------------------------
                //! Syncs all kernels in the current block.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_ACC_NO_CUDA void syncBlockKernels() const
                {
                    // Nothing to do in here because only one thread in a group is allowed.
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
            //! The serial accelerator executor.
            //#############################################################################
            template<
                typename TAcceleratedKernel>
            class KernelExecutorSerial :
                private TAcceleratedKernel,
                private IAcc<AccSerial>
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv, 
                    typename... TKernelConstrArgs>
                ALPAKA_FCT_HOST KernelExecutorSerial(
                    TWorkDiv const & workDiv, 
                    StreamSerial const &,
                    TKernelConstrArgs && ... args) :
                        TAcceleratedKernel(std::forward<TKernelConstrArgs>(args)...),
                        IAcc<AccSerial>()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    (*static_cast<WorkDivSerial *>(this)) = workDiv;
                }
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorSerial(
                    KernelExecutorSerial const & other) :
                        TAcceleratedKernel(other),
                        IAcc<AccSerial>()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    (*static_cast<WorkDivSerial *>(this)) = (*static_cast<WorkDivSerial const *>(&other));
                }
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorSerial(
                    KernelExecutorSerial && other) :
                        TAcceleratedKernel(std::move(other)),
                        IAcc<AccSerial>()
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    (*static_cast<WorkDivSerial *>(this)) = (*static_cast<WorkDivSerial const *>(&other));
                }
#endif
                //-----------------------------------------------------------------------------
                //! Copy assignment.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST KernelExecutorSerial & operator=(KernelExecutorSerial const &) = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL
                ALPAKA_FCT_HOST virtual ~KernelExecutorSerial() = default;
#else
                ALPAKA_FCT_HOST virtual ~KernelExecutorSerial() noexcept = default;
#endif

                //-----------------------------------------------------------------------------
                //! Executes the accelerated kernel.
                //-----------------------------------------------------------------------------
                template<
                    typename... TArgs>
                ALPAKA_FCT_HOST void operator()(
                    TArgs && ... args) const
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    Vec<3u> const v3uiGridBlocksExtents(this->AccSerial::getWorkDiv<Grid, Blocks, dim::Dim3>());
                    Vec<3u> const v3uiBlockKernelsExtents(this->AccSerial::getWorkDiv<Block, Kernels, dim::Dim3>());

                    auto const uiBlockSharedExternMemSizeBytes(BlockSharedExternMemSizeBytes<TAcceleratedKernel>::getBlockSharedExternMemSizeBytes(v3uiBlockKernelsExtents, std::forward<TArgs>(args)...));
                    this->AccSerial::m_vuiExternalSharedMem.reset(
                        new uint8_t[uiBlockSharedExternMemSizeBytes]);

                    // Execute the blocks serially.
                    for(std::uint32_t bz(0); bz<v3uiGridBlocksExtents[2]; ++bz)
                    {
                        this->AccSerial::m_v3uiGridBlockIdx[2] = bz;
                        for(std::uint32_t by(0); by<v3uiGridBlocksExtents[1]; ++by)
                        {
                            this->AccSerial::m_v3uiGridBlockIdx[1] = by;
                            for(std::uint32_t bx(0); bx<v3uiGridBlocksExtents[0]; ++bx)
                            {
                                this->AccSerial::m_v3uiGridBlockIdx[0] = bx;

                                assert(v3uiBlockKernelsExtents[0] == 1);
                                assert(v3uiBlockKernelsExtents[1] == 1);
                                assert(v3uiBlockKernelsExtents[2] == 1);

                                // There is only ever one kernel in a block in the serial accelerator.
                                this->TAcceleratedKernel::operator()(
                                    (*static_cast<IAcc<AccSerial> const *>(this)),
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
            template<
                typename AcceleratedKernel>
            struct AccType<
                serial::detail::KernelExecutorSerial<AcceleratedKernel>>
            {
                using type = AccSerial;
            };
        }
    }

    namespace detail
    {
        //#############################################################################
        //! The serial accelerator kernel executor builder.
        //#############################################################################
        template<
            typename TKernel, 
            typename... TKernelConstrArgs>
        class KernelExecCreator<
            AccSerial, 
            TKernel, 
            TKernelConstrArgs...>
        {
        public:
            using AcceleratedKernel = typename boost::mpl::apply<TKernel, AccSerial>::type;
            using AcceleratedKernelExecutorExtent = KernelExecutorExtent<serial::detail::KernelExecutorSerial<AcceleratedKernel>, TKernelConstrArgs...>;

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
