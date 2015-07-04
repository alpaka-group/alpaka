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
#include <alpaka/workdiv/WorkDivMembers.hpp>    // workdiv::WorkDivMembers
#include <alpaka/idx/gb/IdxGbRef.hpp>           // IdxGbRef
#include <alpaka/idx/bt/IdxBtOmp.hpp>           // IdxBtOmp
#include <alpaka/atomic/AtomicOmpCritSec.hpp>   // AtomicOmpCritSec
#include <alpaka/math/MathStl.hpp>              // MathStl
#include <alpaka/block/shared/BlockSharedAllocMasterSync.hpp>  // BlockSharedAllocMasterSync

// Specialized traits.
#include <alpaka/acc/Traits.hpp>                // AccType
#include <alpaka/exec/Traits.hpp>               // ExecType
#include <alpaka/dev/Traits.hpp>                // DevType
#include <alpaka/size/Traits.hpp>               // size::SizeT

// Implementation details.
#include <alpaka/dev/DevCpu.hpp>                // DevCpu

#include <alpaka/core/OpenMp.hpp>

#include <boost/core/ignore_unused.hpp>         // boost::ignore_unused

#include <memory>                               // std::unique_ptr
#include <typeinfo>                             // typeid

namespace alpaka
{
    namespace exec
    {
        template<
            typename TDim,
            typename TSize>
        class ExecCpuOmp2Threads;

        namespace omp
        {
            namespace omp2
            {
                namespace threads
                {
                    namespace detail
                    {
                        template<
                            typename TDim,
                            typename TSize>
                        class ExecCpuOmp2ThreadsImpl;
                    }
                }
            }
        }
    }
    namespace acc
    {
        namespace omp
        {
            //-----------------------------------------------------------------------------
            //! The CPU OpenMP 2.0 thread accelerator.
            //-----------------------------------------------------------------------------
            namespace omp2
            {
                namespace threads
                {
                    //-----------------------------------------------------------------------------
                    //! The CPU OpenMP 2.0 thread accelerator implementation details.
                    //-----------------------------------------------------------------------------
                    namespace detail
                    {
                        //#############################################################################
                        //! The CPU OpenMP 2.0 thread accelerator.
                        //!
                        //! This accelerator allows parallel kernel execution on a CPU device.
                        //! It uses OpenMP 2.0 to implement the block thread parallelism.
                        //#############################################################################
                        template<
                            typename TDim,
                            typename TSize>
                        class AccCpuOmp2Threads final :
                            public workdiv::WorkDivMembers<TDim, TSize>,
                            public idx::gb::IdxGbRef<TDim, TSize>,
                            public idx::bt::IdxBtOmp<TDim, TSize>,
                            public atomic::AtomicOmpCritSec,
                            public math::MathStl,
                            public block::shared::BlockSharedAllocMasterSync
                        {
                        public:
                            friend class ::alpaka::exec::omp::omp2::threads::detail::ExecCpuOmp2ThreadsImpl<TDim, TSize>;

                        private:
                            //-----------------------------------------------------------------------------
                            //! Constructor.
                            //-----------------------------------------------------------------------------
                            template<
                                typename TWorkDiv>
                            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(
                                TWorkDiv const & workDiv) :
                                    workdiv::WorkDivMembers<TDim, TSize>(workDiv),
                                    idx::gb::IdxGbRef<TDim, TSize>(m_vuiGridBlockIdx),
                                    idx::bt::IdxBtOmp<TDim, TSize>(),
                                    AtomicOmpCritSec(),
                                    math::MathStl(),
                                    block::shared::BlockSharedAllocMasterSync(
                                        [this](){syncBlockThreads();},
                                        [](){return (::omp_get_thread_num() == 0);}),
                                    m_vuiGridBlockIdx(Vec<TDim, TSize>::zeros())
                            {}

                        public:
                            //-----------------------------------------------------------------------------
                            //! Copy constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(AccCpuOmp2Threads const &) = delete;
                            //-----------------------------------------------------------------------------
                            //! Move constructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA AccCpuOmp2Threads(AccCpuOmp2Threads &&) = delete;
                            //-----------------------------------------------------------------------------
                            //! Copy assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Threads const &) -> AccCpuOmp2Threads & = delete;
                            //-----------------------------------------------------------------------------
                            //! Move assignment operator.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA auto operator=(AccCpuOmp2Threads &&) -> AccCpuOmp2Threads & = delete;
                            //-----------------------------------------------------------------------------
                            //! Destructor.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~AccCpuOmp2Threads() = default;

                            //-----------------------------------------------------------------------------
                            //! Syncs all threads in the current block.
                            //-----------------------------------------------------------------------------
                            ALPAKA_FN_ACC_NO_CUDA auto syncBlockThreads() const
                            -> void
                            {
                                // Barrier implementation not waiting for all threads:
                                // http://berenger.eu/blog/copenmp-custom-barrier-a-barrier-for-a-group-of-threads/
                                #pragma omp barrier
                            }

                            //-----------------------------------------------------------------------------
                            //! \return The pointer to the externally allocated block shared memory.
                            //-----------------------------------------------------------------------------
                            template<
                                typename T>
                            ALPAKA_FN_ACC_NO_CUDA auto getBlockSharedExternMem() const
                            -> T *
                            {
                                return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                            }

                        private:
                            // getIdx
                            alignas(16u) Vec<TDim, TSize> mutable m_vuiGridBlockIdx;   //!< The index of the currently executed block.

                            // getBlockSharedExternMem
                            std::unique_ptr<uint8_t, boost::alignment::aligned_delete> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                        };
                    }
                }
            }
        }
    }

    template<
        typename TDim,
        typename TSize>
    using AccCpuOmp2Threads = acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>;

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator accelerator type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct AccType<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
            {
                using type = acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator device properties get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccDevProps<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevCpu const & dev)
                -> alpaka::acc::AccDevProps<TDim, TSize>
                {
                    boost::ignore_unused(dev);

#if ALPAKA_INTEGRATION_TEST
                    auto const uiBlockThreadsCountMax(static_cast<TSize>(4));
#else
                    // m_uiBlockThreadsCountMax
                    // HACK: ::omp_get_max_threads() does not return the real limit of the underlying OpenMP 2.0 runtime:
                    // 'The omp_get_max_threads routine returns the value of the internal control variable, which is used to determine the number of threads that would form the new team,
                    // if an active parallel region without a num_threads clause were to be encountered at that point in the program.'
                    // How to do this correctly? Is there even a way to get the hard limit apart from omp_set_num_threads(high_value) -> omp_get_max_threads()?
                    ::omp_set_num_threads(1024);
                    auto const uiBlockThreadsCountMax(static_cast<TSize>(::omp_get_max_threads()));
#endif
                    return {
                        // m_uiMultiProcessorCount
                        static_cast<TSize>(1),
                        // m_uiBlockThreadsCountMax
                        uiBlockThreadsCountMax,
                        // m_vuiBlockThreadExtentsMax
                        Vec<TDim, TSize>::all(uiBlockThreadsCountMax),
                        // m_vuiGridBlockExtentsMax
                        Vec<TDim, TSize>::all(std::numeric_limits<TSize>::max())};
                }
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator name trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetAccName<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
            {
                ALPAKA_FN_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccCpuOmp2Threads<" + std::to_string(TDim::value) + "," + typeid(TSize).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevType<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator device type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DevManType<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 thread accelerator dimension getter trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
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
            //! The CPU OpenMP 2.0 thread accelerator executor type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct ExecType<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
            {
                using type = exec::ExecCpuOmp2Threads<TDim, TSize>;
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 2.0 thread accelerator size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                acc::omp::omp2::threads::detail::AccCpuOmp2Threads<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
