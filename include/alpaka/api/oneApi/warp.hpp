/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/oneApi/Api.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <algorithm>
#include <cstdint>

#if ALPAKA_LANG_ONEAPI
#    include <sycl/sycl.hpp>

namespace alpaka::onAcc::warp::internal
{
    // GPU back-ends use native SYCL subgroup operations.
    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Activemask::Op<T_Acc, api::OneApi>
    {
        auto operator()(T_Acc const&, api::OneApi) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();

            return getMask(sg);
        }

        static auto getMask(auto const subGroup)
        {
            auto sgMask = sycl::ext::oneapi::group_ballot(subGroup, true);

            constexpr auto const warpSize = T_Acc::getWarpSize();
            using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
            ReturnType mask;
            sgMask.extract_bits(mask, 0u);
            return mask;
        };
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetLaneIdx::Op<T_Acc, api::OneApi>
    {
        constexpr auto operator()(T_Acc const&, api::OneApi) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            // lane id within the warp subgroup
            return sg.get_local_id()[0];
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetNumActiveLanes::Op<T_Acc, api::OneApi>
    {
        constexpr auto operator()(T_Acc const&, api::OneApi) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            // can be less than the warp size
            return sg.get_local_linear_range();
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetWarpIdx::Op<T_Acc, api::OneApi>
    {
        constexpr auto operator()(T_Acc const&, api::OneApi) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            // lane id within the warp subgroup
            return sg.get_group_linear_id();
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetNumWarps::Op<T_Acc, api::OneApi>
    {
        constexpr auto operator()(T_Acc const&, api::OneApi) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            // number of warps within a block
            return sg.get_group_linear_range();
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct All::Op<T_Acc, api::OneApi>
    {
        bool operator()(T_Acc const& acc, api::OneApi, int32_t predicate) const
        {
            using DeviceKind = ALPAKA_TYPEOF(acc[object::deviceKind]);
            if constexpr(DeviceKind{} == alpaka::deviceKind::amdGpu)
            {
                /* Workaround for AMD GPUs: Sycl is taking the results of the non active threads into account
                 * and therefore even if all participating threads have a true predicate the result will be false.
                 * We vote with ballot and mask the result with the active thread mask.
                 */
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                auto activeMask = Activemask::Op<T_Acc, api::OneApi>::getMask(sg);
                auto sgMask = sycl::ext::oneapi::group_ballot(sg, predicate != 0);

                constexpr auto const warpSize = T_Acc::getWarpSize();
                using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
                ReturnType predicateMask;
                sgMask.extract_bits(predicateMask, 0u);
                return activeMask & predicateMask == activeMask;
            }
            else
            {
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                return sycl::all_of_group(sg, predicate != 0);
            }
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Any::Op<T_Acc, api::OneApi>
    {
        bool operator()(T_Acc const& acc, api::OneApi, int32_t predicate) const
        {
            using DeviceKind = ALPAKA_TYPEOF(acc[object::deviceKind]);
            if constexpr(DeviceKind{} == alpaka::deviceKind::amdGpu)
            {
                /* Workaround for AMD GPUs: Sycl is taking the results of non active threads into account
                 * and therefore even if all participating threads have a false predicate the result will be true.
                 * We vote with ballot and mask the result with the active thread mask.
                 */
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                auto activeMask = Activemask::Op<T_Acc, api::OneApi>::getMask(sg);
                auto sgMask = sycl::ext::oneapi::group_ballot(sg, predicate != 0);

                constexpr auto const warpSize = T_Acc::getWarpSize();
                using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
                ReturnType predicateMask;
                sgMask.extract_bits(predicateMask, 0u);
                return activeMask & predicateMask;
            }
            else
            {
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                return sycl::any_of_group(sg, predicate != 0);
            }
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Ballot::Op<T_Acc, api::OneApi>
    {
        auto operator()(T_Acc const&, api::OneApi, int32_t predicate) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto sgMask = sycl::ext::oneapi::group_ballot(sg, predicate != 0);

            constexpr auto const warpSize = T_Acc::getWarpSize();
            using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
            ReturnType mask;
            sgMask.extract_bits(mask, 0u);
            return mask;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct Shfl::Op<T_Acc, api::OneApi, T>
    {
        constexpr T operator()(T_Acc const&, api::OneApi, T const& value, uint32_t srcLane, uint32_t width) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            uint32_t laneIdxInWarp = sg.get_local_id()[0];
            uint32_t partitionOffset = (laneIdxInWarp / width) * width;
            uint32_t srcInPartitionLaneIdx = partitionOffset + (srcLane % width);

            return sycl::select_from_group(sg, value, srcInPartitionLaneIdx);
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflDown::Op<T_Acc, api::OneApi, T>
    {
        constexpr T operator()(T_Acc const&, api::OneApi, T const& value, uint32_t delta, uint32_t width) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();

            uint32_t laneIdxInWarp = sg.get_local_id()[0];
            uint32_t groupEndIdx = (laneIdxInWarp / width + 1) * width;

            T result = sycl::shift_group_left(sg, value, delta);
            if(laneIdxInWarp + delta >= groupEndIdx)
                result = value;
            return result;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflUp::Op<T_Acc, api::OneApi, T>
    {
        constexpr T operator()(T_Acc const&, api::OneApi, T const& value, uint32_t delta, uint32_t width) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();

            uint32_t laneIdxInWarp = sg.get_local_id()[0];
            uint32_t groupStartIdx = (laneIdxInWarp / width) * width;

            T result = sycl::shift_group_right(sg, value, delta);
            if(laneIdxInWarp - groupStartIdx < delta)
                result = value;
            return result;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflXor::Op<T_Acc, api::OneApi, T>
    {
        constexpr T operator()(T_Acc const&, api::OneApi, T const& value, uint32_t laneMask, uint32_t width) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            uint32_t laneIdxInWarp = sg.get_local_id()[0];
            uint32_t groupStartIdx = (laneIdxInWarp / width) * width;
            uint32_t const relativeIdx = laneIdxInWarp - groupStartIdx;
            uint32_t const sourceLane = (relativeIdx % width) ^ laneMask;
            return sycl::select_from_group(sg, value, sourceLane < width ? groupStartIdx + sourceLane : laneIdxInWarp);
        }
    };
} // namespace alpaka::onAcc::warp::internal
#endif
