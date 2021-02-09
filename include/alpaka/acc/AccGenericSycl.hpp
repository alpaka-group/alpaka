/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivGenericSycl.hpp>
#include <alpaka/idx/gb/IdxGbGenericSycl.hpp>
#include <alpaka/idx/bt/IdxBtGenericSycl.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/atomic/AtomicGenericSycl.hpp>
#include <alpaka/intrinsic/IntrinsicGenericSycl.hpp>
#include <alpaka/math/MathGenericSycl.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynGenericSycl.hpp>
#include <alpaka/block/sync/BlockSyncGenericSycl.hpp>
#include <alpaka/warp/WarpGenericSycl.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Sycl.hpp>

#include <cstddef>
#include <string>
#include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! The SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on SYCL devices.
    template<typename TDim, typename TIdx>
    class AccGenericSycl :
        public WorkDivGenericSycl<TDim, TIdx>,
        public gb::IdxGbGenericSycl<TDim, TIdx>,
        public bt::IdxBtGenericSycl<TDim, TIdx>,
        public AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>,
        public math::MathGenericSycl,
        public BlockSharedMemDynGenericSycl,
        public BlockSyncGenericSycl<TDim>,
        public IntrinsicGenericSycl,
        public warp::WarpGenericSycl<TDim>
    {
    public:
#ifdef ALPAKA_SYCL_STREAM_ENABLED
        //-----------------------------------------------------------------------------
        AccGenericSycl(
            Vec<TDim, TIdx> const & threadElemExtent,
            cl::sycl::nd_item<TDim::value> work_item,
            cl::sycl::accessor<std::byte, 1,
                               cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local> shared_acc,
            cl::sycl::stream output_stream) :
                WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item},
                gb::IdxGbGenericSycl<TDim, TIdx>{work_item},
                bt::IdxBtGenericSycl<TDim, TIdx>{work_item},
                AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{},
                math::MathGenericSycl{},
                BlockSharedMemDynGenericSycl{shared_acc},
                BlockSyncGenericSycl<TDim>{work_item},
                IntrinsicGenericSycl{},
                warp::WarpGenericSycl<TDim>{work_item},
                stream{output_stream}
        {}

        cl::sycl::stream stream;
#else
        //-----------------------------------------------------------------------------
        AccGenericSycl(
            Vec<TDim, TIdx> const & threadElemExtent,
            cl::sycl::nd_item<TDim::value> work_item,
            cl::sycl::accessor<std::byte, 1,
                               cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local> shared_acc) :
                WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item},
                gb::IdxGbGenericSycl<TDim, TIdx>{work_item},
                bt::IdxBtGenericSycl<TDim, TIdx>{work_item},
                AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{},
                math::MathGenericSycl{},
                BlockSharedMemDynGenericSycl{shared_acc},
                BlockSyncGenericSycl<TDim>{work_item},
                IntrinsicGenericSycl{},
                warp::WarpGenericSycl<TDim>{work_item}
        {}
#endif

        AccGenericSycl(AccGenericSycl const &) = delete;
        //-----------------------------------------------------------------------------
        AccGenericSycl(AccGenericSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccGenericSycl const &) -> AccGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccGenericSycl &&) -> AccGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        ~AccGenericSycl() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator type trait specialization.
        template<template <typename, typename> typename TAcc, typename TDim, typename TIdx>
        struct AccType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
        {
            using type = TAcc<TDim, TIdx>;
        };

        //#############################################################################
        //! The SYCL accelerator device properties get trait specialization.
        template<template <typename, typename> typename TAcc, typename TDim, typename TIdx>
        struct GetAccDevProps<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(typename DevType<TAcc<TDim, TIdx>>::type const & dev) -> AccDevProps<TDim, TIdx>
            {
                using namespace cl::sycl;

                auto max_threads_dim = dev.m_device.template get_info<info::device::max_work_item_sizes>();
                return {
                    // m_multiProcessorCount
                    alpaka::core::clipCast<TIdx>(dev.m_device.template get_info<info::device::max_compute_units>()),
                    // m_gridBlockExtentMax
                    extent::getExtentVecEnd<TDim>(
                        Vec<DimInt<3u>, TIdx>(
                            // WARNING: There is no SYCL way to determine these values
                            std::numeric_limits<TIdx>::max(),
                            std::numeric_limits<TIdx>::max(),
                            std::numeric_limits<TIdx>::max())),
                    // m_gridBlockCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_blockThreadExtentMax
                    extent::getExtentVecEnd<TDim>(
                        Vec<DimInt<3u>, TIdx>(
                            alpaka::core::clipCast<TIdx>(max_threads_dim[2u]),
                            alpaka::core::clipCast<TIdx>(max_threads_dim[1u]),
                            alpaka::core::clipCast<TIdx>(max_threads_dim[0u]))),
                    // m_blockThreadCountMax
                    alpaka::core::clipCast<TIdx>(
                            dev.m_device.template get_info<info::device::max_work_group_size>()),
                    // m_threadElemExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_threadElemCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_sharedMemSizeBytes
                    dev.m_device.template get_info<info::device::local_mem_size>()
                };
            }
        };

        //#############################################################################
        //! The SYCL accelerator dimension getter trait specialization.
        template<template <typename, typename> typename TAcc, typename TDim, typename TIdx>
        struct DimType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL accelerator idx type trait specialization.
        template<template <typename, typename> typename TAcc, typename TDim, typename TIdx>
        struct IdxType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
        {
            using type = TIdx;
        };
    }
}

#endif
