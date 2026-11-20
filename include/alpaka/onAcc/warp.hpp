/* Copyright 2025 Sergei Bastrakov, David M. Rogers, Bernhard Manfred Gruber, Aurora Perego, Mehmet Yusufoglu, René
 * Widera SPDX-License-Identifier: MPL-2.0
 *
 * Bridges runtime accelerator instances to the trait-based warp intrinsics so kernels can call them without tags.
 * Exposes device-safe `alpaka::onAcc::warp::*` wrappers for ballots, shuffles, and lane queries.
 * Reuses the compile-time warp trait specialisations instead of duplicating backend-specific logic in kernels.
 * Supplies a uniform warp API across CUDA, HIP, SYCL, and host-emulation accelerators.
 *
 * Some example usages:
 * - consteval `alpaka::getWarpSize(api::Cuda{}, deviceKind::NvidiaGpu{})` for tag-driven compile-time logic.
 * - device-side `alpaka::onAcc::warp::getSize(acc)` to query the active warp inside a kernel.
 * - device-side `alpaka::onAcc::warp::shfl(acc, 42, 0u) == 42`
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/internal/warp.hpp"
#include "alpaka/tag.hpp"

#include <algorithm>
#include <cstdint>

namespace alpaka::onAcc::warp
{
    /** Return the bit-mask of active lanes for the warp associated with the accelerator.
     *
     * @return bit mask where the Nth bit is set to 1 if the corresponding thread is participating the call. The return
     * type can be 64bit or 32bit depending on the API.
     */
    template<alpaka::onAcc::concepts::Acc T_Acc>
    constexpr auto activemask(T_Acc const& acc) -> std::conditional_t<T_Acc::getWarpSize() <= 32u, uint32_t, uint64_t>
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Activemask::Op<Acc, Api>{}(acc, Api{});
    }

    /** Return the lane index of the current thread within its warp. */
    constexpr uint32_t getLaneIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        return internal::getLaneIdx(acc);
    }

    /** Return the warp index within the block. */
    constexpr uint32_t getWarpIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        return internal::getWarpIdx(acc);
    }

    /** Evaluates predicate for all active threads of the warp
     *
     * It follows the logic of __all_sync(__activemask(), predicate) in CUDA but returns a boolean.
     *
     * Note:
     * * The programmer must ensure that all threads calling this function are executing
     *   the same line of code. In particular, it is not portable to write
     *   if(a) {all} else {all}.
     *
     * @param predicate The predicate value for current thread.
     * @return true if all threads predicate non zero, else false
     */
    constexpr bool all(alpaka::onAcc::concepts::Acc auto const& acc, int32_t predicate)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::All::Op<Acc, Api>{}(acc, Api{}, predicate);
    }

    /** Evaluates predicate for all active threads of the warp.
     *
     * It follows the logic of __any_sync(__activemask(), predicate) in CUDA but returns a boolean.
     *
     * Note:
     * * The programmer must ensure that all threads calling this function are executing
     *   the same line of code. In particular, it is not portable to write
     *   if(a) {any} else {any}.
     *
     * @param predicate The predicate value for current thread.
     * @return true if at least one threads predicate is non zero, else false
     */
    constexpr bool any(alpaka::onAcc::concepts::Acc auto const& acc, int32_t predicate)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Any::Op<Acc, Api>{}(acc, Api{}, predicate);
    }

    /** Evaluates predicate for all non-exited threads in a warp and returns
     * a 32- or 64-bit unsigned integer (depending on the accelerator)
     * whose Nth bit is set if and only if predicate evaluates to non-zero
     * for the Nth thread of the warp and the Nth thread is active.
     *
     * It follows the logic of __ballot_sync(__activemask(), predicate) in CUDA.
     *
     * Note:
     * * The programmer must ensure that all threads calling this function are executing
     *   the same line of code. In particular, it is not portable to write
     *   if(a) {ballot} else {ballot}.
     *
     * @param predicate The predicate value for current thread.
     * @return bit mask where the Nth bit is set to 1 if the corresponding threads predicate was non zero. The return
     * type can be 64bit or 32bit depending on the API.
     */
    template<alpaka::onAcc::concepts::Acc T_Acc>
    constexpr auto ballot(T_Acc const& acc, int32_t predicate)
        -> std::conditional_t<T_Acc::getWarpSize() <= 32u, uint32_t, uint64_t>
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Ballot::Op<Acc, Api>{}(acc, Api{}, predicate);
    }

    /** Return the warp size.
     *
     * A warp is a collection of threads which work in lock step (executing the same command).
     * The warp size can be larger than the number of threads executed in the kernel/ thread block.
     * @{
     */
    template<concepts::Acc T_Acc>
    constexpr uint32_t getSize()
    {
        return internal::getSize<T_Acc>();
    }

    template<concepts::Acc T_Acc>
    constexpr uint32_t getSize(T_Acc const&)
    {
        return T_Acc::getWarpSize();
    }

    /** @} */

    /** Exchange data between threads within a warp.
     *
     * Effectively executes:
     *
     *     __shared__ int32_t values[warpsize];
     *     values[threadIdx.x] = value;
     *     __syncthreads();
     *     return values[width*(threadIdx.x/width) + srcLane%width];
     *
     * However, it does not use shared memory.
     *
     *  Commonly used with width = warpsize (the default), (returns values[srcLane])
     *
     * This method supports to be called in diverging control flow branches if you only query values from threads
     * within the same branch path.
     *
     * @param value value to broadcast, only used if other thread is addressing the lane of this thread.
     * @param srcLane source lane index within the group range [0; width).
     * @param width number of threads receiving a single value, must be a power of 2.
     * @return val from the thread index srcLane.
     */
    template<typename T, alpaka::onAcc::concepts::Acc T_Acc>
    requires(std::is_trivially_copyable_v<T>)
    constexpr T shfl(T_Acc const& acc, T const& value, uint32_t srcLane, uint32_t width = getSize<T_Acc>())
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Shfl::Op<Acc, Api, T>{}(acc, Api{}, value, srcLane, width != 0u ? width : getSize<T_Acc>());
    }

    /** Read data from threads with higher lane index within a warp.
     *
     * It copies from a lane with higher ID relative to caller.
     * The lane ID is calculated by adding delta to the caller's lane ID.
     *
     * Effectively executes:
     *
     *     __shared__ int32_t values[warpsize];
     *     values[threadIdx.x] = value;
     *     __syncthreads();
     *     return (threadIdx.x % width + delta < width) ? values[threadIdx.x + delta] : values[threadIdx.x];
     *
     * However, it does not use shared memory.
     *
     * Notes:
     * * The programmer must ensure that all threads calling this
     *   function (and the srcLane) are executing the same line of code.
     *   In particular it is not portable to write if(a) {shfl} else {shfl}.
     *
     * * Commonly used with width = warpsize (the default), (returns values[threadIdx.x+delta] if threadIdx.x+delta <
     * warpsize)
     *
     * @param value value to broadcast
     * @param delta corresponds to the delta used to compute the lane ID
     * @param width size of the group participating in the shuffle operation, must be a power of 2.
     * @return the value from the thread index lane ID + delta within the group build by width, else value.
     */
    template<typename T, alpaka::onAcc::concepts::Acc T_Acc>
    requires(std::is_trivially_copyable_v<T>)
    constexpr T shflDown(T_Acc const& acc, T const& value, uint32_t delta, uint32_t width = getSize<T_Acc>())
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::ShflDown::Op<Acc, Api, T>{}(acc, Api{}, value, delta, width != 0u ? width : getSize<T_Acc>());
    }

    /** Read data from threads with lower lane index within a warp.
     *
     * It copies from a lane with lower ID relative to caller.
     * The lane ID is calculated by subtracting delta from the caller's lane ID.
     *
     * Effectively executes:
     *
     *     __shared__ int32_t values[warpsize];
     *     values[threadIdx.x] = value;
     *     __syncthreads();
     *     return (threadIdx.x % width >= delta) ? values[threadIdx.x - delta] : values[threadIdx.x];
     *
     * However, it does not use shared memory.
     *
     * Notes:
     * * The programmer must ensure that all threads calling this
     *   function (and the srcLane) are executing the same line of code.
     *   In particular it is not portable to write if(a) {shfl} else {shfl}.
     *
     * Commonly used with width = warpsize (the default), (returns values[threadIdx.x - delta] if threadIdx.x >= delta)
     *
     * @param value value to broadcast
     * @param delta corresponds to the delta used to compute the lane ID
     * @param width size of the group participating in the shuffle operation, must be a power of 2.
     * @return the value from the thread index lane ID + delta within the group build by width, else value.
     */
    template<typename T, alpaka::onAcc::concepts::Acc T_Acc>
    requires(std::is_trivially_copyable_v<T>)
    constexpr T shflUp(T_Acc const& acc, T const& value, uint32_t delta, uint32_t width = getSize<T_Acc>())
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::ShflUp::Op<Acc, Api, T>{}(acc, Api{}, value, delta, width != 0u ? width : getSize<T_Acc>());
    }

    /** Exchange data between threads within a warp.
     *
     * It copies from a lane based on bitwise XOR of own lane ID.
     * The lane ID is calculated by performing a bitwise XOR of the caller's lane ID with laneMask
     *
     * Effectively executes:
     *
     *     __shared__ int32_t values[warpsize];
     *     values[threadIdx.x] = value;
     *     __syncthreads();
     *     int lane = threadIdx.x ^ laneMask;
     *     return values[lane / width > threadIdx.x / width ? threadIdx.x : lane];
     *
     * However, it does not use shared memory.
     *
     * Notes:
     * * The programmer must ensure that all threads calling this
     *   function (and the srcLane) are executing the same line of code.
     *   In particular it is not portable to write if(a) {shfl} else {shfl}.
     *
     * * Commonly used with width = warpsize (the default), (returns values[threadIdx.x^laneMask])
     *
     * * Width must be a power of 2.
     * @param value value to broadcast
     * @param laneMask mask applied to the thread lane index within the subgroup created by width.
     * @param width size of the group participating in the shuffle operation, must be a power of 2.
     * @return the value from the thread index lane ID
     */
    template<typename T, alpaka::onAcc::concepts::Acc T_Acc>
    requires(std::is_trivially_copyable_v<T>)
    constexpr T shflXor(T_Acc const& acc, T const& value, uint32_t laneMask, uint32_t width = getSize<T_Acc>())
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::ShflXor::Op<Acc, Api, T>{}(
            acc,
            Api{},
            value,
            laneMask,
            width != 0u ? width : getSize<T_Acc>());
    }

    /** Reduce a value over the lanes of a warp (butterfly with shflXor).
     *
     * Collective: all lanes of the warp must reach the same call in converged control flow. A branch around the call
     * is only allowed if all lanes of the warp take it, otherwise select the value inside the branch and call
     * reduce() with all lanes after it.
     * No block synchronization and no memory fence is performed.
     *
     * @attention The number of threads of the block must be a multiple of the warp size or a power of two smaller
     * than the warp size.
     *
     * @param value contribution of this lane
     * @param op associative and commutative binary functor
     * @return reduction over the warp, identical in all lanes
     */
    template<typename T, alpaka::onAcc::concepts::Acc T_Acc>
    requires(std::is_trivially_copyable_v<T>)
    constexpr T reduce(T_Acc const& acc, T value, auto const& op)
    {
        // a block can be smaller than a warp
        auto const numThreadsInBlock = static_cast<uint32_t>(acc[alpaka::layer::thread].count().product());
        uint32_t const numLanes = std::min(getSize<T_Acc>(), numThreadsInBlock);
        for(uint32_t laneMask = numLanes / 2u; laneMask > 0u; laneMask /= 2u)
            value = op(value, shflXor(acc, value, laneMask, numLanes));
        return value;
    }
} // namespace alpaka::onAcc::warp
