/* Copyright 2025 Andrea Bocci, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/elem/Traits.hpp"
#include "alpaka/exec/UniformElements.hpp"
#include "alpaka/extent/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/intrinsic/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/warp/Traits.hpp"
#include "alpaka/workdiv/WorkDivHelpers.hpp"

#include <iterator>
#include <type_traits>

namespace alpaka
{

    namespace detail
    {

        /**
         * @brief Compute lane mask
         *
         * @param mask Input lane index in the warp
         *
         * @return compute lane mask:
         */
        template<concepts::Acc TAcc>
        requires(
            std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuCudaRt>
            || std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuHipRt>)
        ALPAKA_FN_ACC inline constexpr auto getLaneMask(std::uint32_t const lane_idx) -> typename TAcc::mask_type
        {
            return (1ULL << lane_idx);
        }

        template<concepts::Acc TAcc>
        requires std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>
        ALPAKA_FN_ACC inline constexpr auto getLaneMask(std::uint32_t const lane_idx) -> typename TAcc::mask_type
        {
            return (1U << lane_idx);
        }

        /**
         * @brief Check that given lane is active in the custom lane mask
         *
         * @param mask Input mask.
         * @param mask Input lane index in the warp
         *
         * @return True if active, otherwise false.
         */
        template<concepts::Acc TAcc>
        ALPAKA_FN_ACC inline constexpr bool isWorkLane(
            typename TAcc::mask_type const work_mask,
            std::uint32_t const lane_idx)
        {
            return ((work_mask >> lane_idx) & 1);
        }

        /**
         * @brief Returns the position of the least significant set bit in a mask.
         *
         * @tparam TAcc Alpaka accelerator type.
         *
         * @param acc   Alpaka accelerator instance.
         * @param mask  Input mask.
         *
         * @return Index of least significant 1 bit (0-based). (or warp size if x == 0).
         */
        template<concepts::Acc TAcc>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE auto getLs1bIdx(TAcc const& acc, typename TAcc::mask_type const mask) ->
            typename TAcc::mask_type
        {
            using mask_t = typename TAcc::mask_type;

            if(mask == 0)
                return static_cast<mask_t>(warp::getSizeCompileTime<TAcc>());

            if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
                return 0;

            using signed_warp_mask_t = std::make_signed_t<mask_t>;

            auto const pos = alpaka::ffs(acc, static_cast<signed_warp_mask_t>(mask));
            return static_cast<mask_t>(pos - 1);
        }

        /**
         * @brief Returns true if a given lane is represented in the least significant bit in a mask.
         *
         * @tparam TAcc Alpaka accelerator type.
         *
         * @param acc   Alpaka accelerator instance.
         * @param mask  Input mask.
         * @param lane_idx Current thread's lane index.
         *
         * @return True if lane_idx is the least significant bit in a mask, otherwise faulse .
         */
        template<concepts::Acc TAcc>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE bool isLs1bIdx(typename TAcc::mask_type const mask, uint32_t const lane_idx)
        {
            using mask_t = typename TAcc::mask_type;

            if(mask == 0)
                return false;

            if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
                return true;

            mask_t const lane_mask = getLaneMask<TAcc>(lane_idx);

            // First check wether the lane is represented at all, otherwise check the trivial case:
            if((mask & lane_mask) == 0)
                return false;
            else if(lane_idx == 0)
                return true;

            constexpr std::uint32_t w_extent = warp::getSizeCompileTime<TAcc>();

            mask_t const inverted_mask = mask << (w_extent - lane_idx);

            return (inverted_mask == 0);
        }

        /**
         * @brief Performs warp-level exclusive prefix sum
         *
         * @tparam TAcc Alpaka accelerator type.
         * @tparam accum If true, broadcast total accumulated value to lowest active lane.
         *
         * @param acc   Alpaka accelerator instance.
         * @param val   Value to include in the prefix sum.
         * @param lane_idx Current thread's lane index.
         *
         * @return Exclusive prefix sum value for the current lane.
         * convention used here:
         * - lanes 1..(w_extent-1) receive the exclusive prefix sum (CSR offsets within the warp),
         * - lane 0 receives the total sum over the warp (used as the per-warp NNZ aggregate)
         */

        template<concepts::Acc TAcc, bool keep_total_sum = true>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE std::uint32_t warpExclusiveSum(
            TAcc const& acc,
            std::uint32_t const val,
            std::uint32_t const lane_idx)
        {
            using mask_t = typename TAcc::mask_type;

            if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
                return keep_total_sum ? val : 0;

            constexpr std::uint32_t w_extent = warp::getSizeCompileTime<TAcc>();

            std::uint32_t local_offset = val;

            // Do inclusive sum first:
            for(std::uint32_t step = 1; step < w_extent; step *= 2)
            {
                auto const res = warp::shfl_up(acc, local_offset, step, w_extent);
                if(lane_idx >= step)
                    local_offset += res;
            }

            if constexpr(keep_total_sum)
            {
                std::uint32_t const high_lane_idx = w_extent - 1;

                // send last lane value (total tile offset) to lane idx = low_lane_idx:
                mask_t const active_mask = 1 | getLaneMask<TAcc>(high_lane_idx);
                if(isWorkLane<TAcc>(active_mask, lane_idx))
                {
                    std::uint32_t const tmp = warp::shfl(acc, active_mask, local_offset, high_lane_idx, w_extent);

                    if(lane_idx == 0)
                        local_offset = tmp; // lane 0 keeps full (inclusive for the last lane) sum
                }
            }
            return lane_idx == 0 ? local_offset : local_offset - val; // we return exclusive sum!
        }

        /**
         * @brief Returns logical index for a given physical lane index based on custom lane mask.
         *
         * @tparam TAcc Alpaka accelerator type.
         *
         * @param acc Alpaka accelerator instance.
         * @param mask Input bitmask.
         * @param lane_idx imput phys. lane index
         *
         * @return Index of the lane in the mask
         */

        template<concepts::Acc TAcc>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE std::uint32_t getLogicalLaneIdx(
            TAcc const& acc,
            typename TAcc::mask_type const mask,
            std::uint32_t const lane_idx)
        {
            if(lane_idx == 0)
                return lane_idx; // nothing to do, phys idx coincide with the logical one.
            auto const lane_mask = mask & (getLaneMask<TAcc>(lane_idx) - 1);
            return static_cast<std::uint32_t>(alpaka::popcount(acc, lane_mask)); // Count 1s below current lane
        }

        /**
         * @brief Returns physical lane index for a given logical lane index based on custom lane mask.
         *
         * @tparam TAcc Alpaka accelerator type.
         *
         * @param acc Alpaka accelerator instance.
         * @param mask Input mask.
         * @param logical_lane_idx input logical lane index
         *
         * @return Physical index of the lane in the mask
         */

        template<concepts::Acc TAcc>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE std::uint32_t getPhysicalLaneIdx(
            TAcc const& acc,
            typename TAcc::mask_type const mask,
            std::int32_t logical_lane_idx)
        {
            using signed_warp_mask_t = std::make_signed_t<typename TAcc::mask_type>;

            if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
                return 0;

            signed_warp_mask_t m = static_cast<signed_warp_mask_t>(mask);

            while(logical_lane_idx--)
                m &= (m - 1);

            auto const pos = alpaka::ffs(acc, m);

            return static_cast<std::uint32_t>(pos - 1);
        }

        /**
         * @brief generic warp reduction
         *
         * @tparam TAcc Alpaka accelerator type.
         *
         * @param acc Alpaka accelerator instance.
         * @param in input value to reduce
         * @param f reducer
         *
         * @return return reduced value (propagated to all lanes in the mask by default)
         */

        template<concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
        requires(
            std::is_arithmetic_v<reduce_t>
            && (std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuCudaRt>
                || std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagGpuHipRt>) )
        ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpReduce(TAcc const& acc, reduce_t const in, reducer_t const f)
            -> reduce_t
        {
            constexpr std::uint32_t w_extent = warp::getSizeCompileTime<TAcc>();

            reduce_t result = in;

            for(std::uint32_t offset = w_extent / 2; offset > 0; offset /= 2)
            {
                result = f(result, warp::shfl_down(acc, result, offset, w_extent));
            }

            if constexpr(all)
                result = warp::shfl(acc, result, 0, w_extent);

            return result;
        }

        template<concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
        requires(std::is_arithmetic_v<reduce_t> && std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
        ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpReduce(TAcc const&, reduce_t const in, reducer_t const) -> reduce_t
        {
            reduce_t result = in;

            return result;
        }

        /**
         * @brief Sparse warp reduction
         *
         * @tparam TAcc Alpaka accelerator type.
         *
         * @param acc Alpaka accelerator instance
         * @param mask input mask
         * @param in input value to reduce
         * @param f reducer
         *
         * @return return reduced value (propagated to all lanes in the mask by default)
         */

        template<concepts::Acc TAcc, typename reduce_t, typename reducer_t, bool all = true>
        requires std::is_arithmetic_v<reduce_t>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpSparseReduce(
            TAcc const& acc,
            typename TAcc::mask_type const mask,
            std::uint32_t const lane_idx,
            reduce_t const in,
            reducer_t const f) -> reduce_t
        {
            constexpr std::uint32_t w_extent = warp::getSizeCompileTime<TAcc>();

            if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
                return mask == 0 ? 0 : in;

            // Non-active lanes should skip the reduction:
            if(isWorkLane<TAcc>(mask, lane_idx) == false)
                return in;

            std::uint32_t nActiveLanes
                = static_cast<std::uint32_t>(alpaka::popcount(acc, mask)); // count number of active lanes

            // First check if this is just a single active lane in the warp:
            if(nActiveLanes == 1)
                return in;

            // Compute the next power of two:
            std::uint32_t const pow2 = w_extent - static_cast<std::uint32_t>(alpaka::clz(acc, nActiveLanes - 1));
            std::uint32_t const pow2_boundary = 1 << pow2;

            std::uint32_t const logical_lane_idx = getLogicalLaneIdx(acc, mask, lane_idx);

            reduce_t res = in;

            for(std::uint32_t offset = pow2_boundary / 2; offset > 0; offset /= 2)
            {
                std::uint32_t const logical_src_lane_idx = logical_lane_idx + offset;
                std::uint32_t const src_lane_idx
                    = (logical_src_lane_idx < nActiveLanes)
                          ? getPhysicalLaneIdx(acc, mask, static_cast<std::int32_t>(logical_src_lane_idx))
                          : lane_idx;

                reduce_t const neigh_res
                    = warp::shfl(acc, mask, res, static_cast<std::int32_t>(src_lane_idx), w_extent);

                if(logical_src_lane_idx < nActiveLanes)
                    res = f(res, neigh_res);

                warp::syncWarpThreads(acc, mask);
            }

            if constexpr(all)
            {
                auto const low_lane_idx = getPhysicalLaneIdx(acc, mask, 0);
                res = warp::shfl(acc, mask, res, static_cast<std::int32_t>(low_lane_idx), w_extent);
            }

            return res;
        }

        /**
         * @brief Performs warp-level sparse exclusive prefix sum (masked version of warpExclusiveSum, see above )
         *
         * @tparam TAcc Alpaka accelerator type.
         * @tparam accum If true, broadcast total accumulated value to lowest active lane.
         *
         * @param acc   Alpaka accelerator instance.
         * @param mask  input mask
         * @param val   Value to include in the prefix sum.
         * @param lane_idx Current thread's lane index.
         *
         * @return Exclusive prefix sum value for the current lane.
         */

        template<concepts::Acc TAcc, bool keep_total_sum = true>
        ALPAKA_FN_ACC ALPAKA_FN_INLINE auto warpSparseExclusiveSum(
            TAcc const& acc,
            typename TAcc::mask_type const mask,
            std::uint32_t const val,
            std::uint32_t const lane_idx) -> std::uint32_t
        {
            constexpr std::uint32_t w_extent = warp::getSizeCompileTime<TAcc>();

            if constexpr(std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>)
                return keep_total_sum == false ? 0 : (mask == 0 ? 0 : val);

            // Non-active lanes should skip the reduction:
            if(isWorkLane<TAcc>(mask, lane_idx) == false)
                return 0;

            // count number of active lanes
            std::uint32_t const nActiveLanes = static_cast<std::uint32_t>(alpaka::popcount(acc, mask));
            // First check if this is just a single active lane in the warp:
            if(nActiveLanes == 1)
                return val; // nothing to do, note that this is the inclusive "sum": low lane always keeps the whole
                            // sum

            // Compute the next power of two:
            std::uint32_t const pow2 = w_extent - static_cast<std::uint32_t>(alpaka::clz(acc, nActiveLanes - 1));
            std::uint32_t const pow2_boundary = 1 << pow2;

            std::uint32_t const logical_lane_idx = getLogicalLaneIdx(acc, mask, lane_idx);

            std::uint32_t local_offset = val;

            for(std::uint32_t step = 1; step < pow2_boundary; step *= 2)
            {
                std::uint32_t const src_lane_idx
                    = (logical_lane_idx >= step)
                          ? getPhysicalLaneIdx(acc, mask, static_cast<std::int32_t>(logical_lane_idx - step))
                          : lane_idx;
                std::uint32_t const tmp_val
                    = warp::shfl(acc, mask, local_offset, static_cast<std::int32_t>(src_lane_idx), w_extent);

                if(logical_lane_idx >= step)
                    local_offset += tmp_val;
            }

            if constexpr(keep_total_sum)
            {
                std::uint32_t const high_lane_idx
                    = getPhysicalLaneIdx(acc, mask, static_cast<std::int32_t>(nActiveLanes - 1));
                // send last lane value (total tile offset) to lane idx = low_lane_idx:
                std::uint32_t const tmp
                    = warp::shfl(acc, mask, local_offset, static_cast<std::int32_t>(high_lane_idx), w_extent);

                if(logical_lane_idx == 0)
                    local_offset = tmp; // lane 0 keeps full (inclusive for the last lane) sum
            }
            return logical_lane_idx == 0
                       ? local_offset
                       : local_offset
                             - val; // we return exclusive sum, except zero logical lane (which returns total offset)
        }


    } // namespace detail

} // namespace alpaka
