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
#include <alpaka/accs/omp/omp2/Idx.hpp>     // IdxOmp2
#include <alpaka/accs/omp/omp2/Atomic.hpp>  // AtomicOmp2

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/accs/omp/Common.hpp>
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu
#include <alpaka/devs/cpu/Event.hpp>        // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>       // StreamCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <memory>                           // std::unique_ptr
#include <vector>                           // std::vector

namespace alpaka
{
    namespace accs
    {
        namespace omp
        {
            //-----------------------------------------------------------------------------
            //! The OpenMP2 accelerator.
            //-----------------------------------------------------------------------------
            namespace omp2
            {
                //-----------------------------------------------------------------------------
                //! The OpenMP2 accelerator implementation details.
                //-----------------------------------------------------------------------------
                namespace detail
                {
                    class ExecOmp2;

                    //#############################################################################
                    //! The OpenMP2 accelerator.
                    //!
                    //! This accelerator allows parallel kernel execution on a cpu device.
                    // \TODO: Offloading?
                    //! It uses OpenMP2 to implement the parallelism.
                    //#############################################################################
                    class AccOmp2 :
                        protected alpaka::workdiv::BasicWorkDiv,
                        protected IdxOmp2,
                        protected AtomicOmp2
                    {
                    public:
                        friend class ::alpaka::accs::omp::omp2::detail::ExecOmp2;

                    private:
                        //-----------------------------------------------------------------------------
                        //! Constructor.
                        //-----------------------------------------------------------------------------
                        template<
                            typename TWorkDiv>
                        ALPAKA_FCT_ACC_NO_CUDA AccOmp2(
                            TWorkDiv const & workDiv) :
                                alpaka::workdiv::BasicWorkDiv(workDiv),
                                IdxOmp2(m_v3uiGridBlockIdx),
                                AtomicOmp2(),
                                m_v3uiGridBlockIdx(Vec3<>::zeros())
                        {}

                    public:
                        //-----------------------------------------------------------------------------
                        //! Copy constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA AccOmp2(AccOmp2 const &) = delete;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                        //-----------------------------------------------------------------------------
                        //! Move constructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA AccOmp2(AccOmp2 &&) = delete;
#endif
                        //-----------------------------------------------------------------------------
                        //! Copy assignment.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccOmp2 const &) -> AccOmp2 & = delete;
                        //-----------------------------------------------------------------------------
                        //! Destructor.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA virtual ~AccOmp2() noexcept = default;

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
                                *static_cast<IdxOmp2 const *>(this),
                                *static_cast<workdiv::BasicWorkDiv const *>(this));
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
                                *static_cast<workdiv::BasicWorkDiv const *>(this));
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
                                *static_cast<AtomicOmp2 const *>(this));
                        }

                        //-----------------------------------------------------------------------------
                        //! Syncs all threads in the current block.
                        //-----------------------------------------------------------------------------
                        ALPAKA_FCT_ACC_NO_CUDA auto syncBlockThreads() const
                        -> void
                        {
                            #pragma omp barrier
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
                }
            }
        }
    }

    using AccOmp2 = accs::omp::omp2::detail::AccOmp2;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The OpenMP2 accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::AccOmp2;
            };
            //#############################################################################
            //! The OpenMP2 accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetAccDevProps<
                accs::omp::omp2::detail::AccOmp2>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::detail::DevCpu const & dev)
                -> alpaka::acc::AccDevProps
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST

                    UInt const uiBlockThreadsCountMax(4u);
#else
                    // m_uiBlockThreadsCountMax
                    // HACK: ::omp_get_max_threads() does not return the real limit of the underlying OpenMP2 runtime:
                    // 'The omp_get_max_threads routine returns the value of the internal control variable, which is used to determine the number of threads that would form the new team,
                    // if an active parallel region without a num_threads clause were to be encountered at that point in the program.'
                    // How to do this correctly? Is there even a way to get the hard limit apart from omp_set_num_threads(high_value) -> omp_get_max_threads()?
                    ::omp_set_num_threads(1024);
                    UInt const uiBlockThreadsCountMax(static_cast<UInt>(::omp_get_max_threads()));
#endif
                    return alpaka::acc::AccDevProps(
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_v3uiBlockThreadExtentsMax
                        Vec3<>::all(uiBlockThreadsCountMax),
                        // m_v3uiGridBlockExtentsMax
                        Vec3<>::all(std::numeric_limits<Vec3<>::Val>::max()));
                }
            };
            //#############################################################################
            //! The OpenMP2 accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::omp::omp2::detail::AccOmp2>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccOmp2";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The OpenMP2 accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The OpenMP2 accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = accs::omp::omp2::detail::ExecOmp2;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The OpenMP2 accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The OpenMP2 accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The OpenMP2 accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::omp::omp2::detail::AccOmp2>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
        }
    }
}
