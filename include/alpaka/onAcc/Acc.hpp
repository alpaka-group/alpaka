/* Copyright 2024 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/MdSpanArray.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/interface.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onAcc
{
    template<typename T_Storage>
    struct Acc : T_Storage
    {
        constexpr Acc(T_Storage const& storage) : T_Storage{storage}
        {
        }

        constexpr Acc(Acc const&) = delete;
        constexpr Acc(Acc&&) = delete;
        constexpr Acc& operator=(Acc const&) = delete;
        constexpr Acc& operator=(Acc&&) = delete;

        /** Get the n-dimensional indices within the origin in the quantity of the selected unit
         *
         * @attention if origin or unit is warp/s a one dimensional vector is returned
         */
        constexpr alpaka::concepts::Vector auto getIdxWithin(concepts::Origin auto origin, concepts::Unit auto unit)
            const
        {
            return internalCompute::GetIdxWithin::Op<Acc, ALPAKA_TYPEOF(origin), ALPAKA_TYPEOF(unit)>{}(
                *this,
                origin,
                unit);
        }

        /** Get the n-dimensional extents of an origin in the quantity of the selected unit
         *
         * @attention if origin or unit is warp/s a one dimensional vector is returned
         */
        constexpr alpaka::concepts::Vector auto getExtentsOf(concepts::Origin auto origin, concepts::Unit auto unit)
            const
        {
            return internalCompute::GetExtentsOf::Op<Acc, ALPAKA_TYPEOF(origin), ALPAKA_TYPEOF(unit)>{}(
                *this,
                origin,
                unit);
        }

        static constexpr bool hasKey(auto key)
        {
            constexpr auto idx = alpaka::internal::KeyIdx<ALPAKA_TYPEOF(key), std::decay_t<T_Storage>>::value;
            return idx != -1;
        }

        constexpr auto getApi() const
        {
            return T_Storage::operator[](object::api);
        }

        constexpr auto getDeviceKind() const
        {
            return T_Storage::operator[](object::deviceKind);
        }

        constexpr auto getExecutor() const
        {
            return T_Storage::operator[](object::exec);
        }

        /** Check if a frame spec was used to enqueue the kernel
         *
         * To be able to use the result as constexpr value you must call the static function via the type
         * `Acc_t::launchedWithFrameSpec()` else the warp size can only be used as runtime value.
         *
         * If the kernel was enqueued via a frame specification the thread specification to within the kernel can
         * differ, the number of thread block is not necessarily equal to the number of frames and the thread block
         * extent is not necessarily equal to the frame extents.
         *
         * @return true if the kernel was launched based on a frame specification, false if a thread specification was
         * used.
         */
        static constexpr bool launchedWithFrameSpec()
        {
            // is std::true_type or std::false_type
            return ALPAKA_TYPEOF(std::declval<T_Storage>()[object::launchedWidthFrameSpec])::value;
        }

        /** Get the warp size
         *
         * To be able to use the warp size as constexpr value you must call the static function via the type
         * `Acc_t::getWarpSize()` else the warp size can only be used as runtime value.
         * Alternative you can use the free function onAcc::warp::getSize<Acc_t>();
         */
        static constexpr uint32_t getWarpSize()
        {
            constexpr uint32_t w = ALPAKA_TYPEOF(std::declval<T_Storage>()[object::warpSize])::value;
            return w;
        }
    };

    namespace concepts
    {
        /** Concept to check if a type is an accelerator
         *
         * @tparam T_Acc Type to check
         * @tparam T_Api Enforce an API type, if not provided api type is not checked
         */
        template<typename T_Acc, typename T_Api = alpaka::NotRequired>
        concept Acc = alpaka::concepts::SpecializationOf<T_Acc, onAcc::Acc>
                      && (std::same_as<T_Api, ALPAKA_TYPEOF(std::declval<T_Acc>().getApi())>
                          || std::same_as<T_Api, alpaka::NotRequired>);
    } // namespace concepts

    /** Synchronize all threads within a given scope */
    template<alpaka::concepts::Layer T_Scope>
    constexpr void sync(concepts::Acc auto const& acc, T_Scope scope)
    {
        internalCompute::sync(acc, scope);
    }

    /** Synchronize all threads within a thread block */
    constexpr void syncBlockThreads(concepts::Acc auto const& acc)
    {
        internalCompute::sync(acc, alpaka::layer::block);
    }

    /** Create a variable located in the thread blocks shared memory
     *
     * @code{.cpp}
     * // creates a reference to a float value
     * auto& foo = declareSharedVar<float, uniqueId()>(acc);
     * @endcode
     *
     * @attention The data is not initialized; it can contain garbage.
     *
     * @tparam T The type that should be created; the constructor is not called
     * @tparam T_uniqueId ID that is unique inside a kernel.
     *                  Reusing the id will return the same memory declared before with the same id.
     * @return Result should be taken by reference
     */
    template<typename T, size_t T_uniqueId>
    constexpr decltype(auto) declareSharedVar(concepts::Acc auto const& acc)
    {
        return internalCompute::declareSharedVar<T, T_uniqueId>(acc);
    }

    /** creates an M-dimensional array
     *
     * @code{.cpp}
     * // creates a MdSpan view to a float value, do NOT use a reference here
     * auto fooArrayMd = declareSharedVar<float, uniqueId()>(acc, CVec<uint32_t, 5, 8>{});
     * @endcode
     *
     * @attention The data is not initialized it can contain garbage.
     *
     * @tparam T type which should be created, the constructor is not called
     * @tparam T_uniqueId id which is unique inside a kernel.
     *                  Reusing the id will return the same memory declared before with the same id.
     * @param extent M-dimensional extent in elements for each dimension, 1 - M dimensions are supported
     * @return MdSpan non owning view to the corresponding data, you should NOT store a reference to the handle
     */
    template<typename T, size_t T_uniqueId>
    constexpr decltype(auto) declareSharedMdArray(
        concepts::Acc auto const& acc,
        alpaka::concepts::CVector auto const& extent)
    {
        using CArrayType = typename CArrayType<T, ALPAKA_TYPEOF(extent)>::type;
        /* XOR with hash to avoid issues in case the user is using the same id to create an array and normal shared
         * variables.
         */
        constexpr size_t id = T_uniqueId ^ 0x9e37'79b9'7f4a'7c15;
        constexpr auto alignment = Alignment<alignof(T)>{};
        return MdSpanArray<CArrayType, typename ALPAKA_TYPEOF(extent)::value_type, ALPAKA_TYPEOF(alignment)>{
            declareSharedVar<CArrayType, id>(acc)};
    }

    /** Get block shared dynamic memory.
     *
     * The available size of the memory can be defined by specializing 'onHost::trait::GetDynSharedMemBytes' or adding
     * a public member variable 'uint32_t dynSharedMemBytes' for a kernel. The Memory can be accessed by all threads
     * within a block. Access to the memory is not thread safe.
     *
     * \tparam T The element type.
     * \return Pointer to pre-allocated contiguous memory.
     */
    template<typename T>
    constexpr auto getDynSharedMem(concepts::Acc auto const& acc) -> T*
    {
        return internalCompute::declareDynamicSharedMem<T>(acc);
    }

} // namespace alpaka::onAcc

namespace alpaka::onAcc::internalCompute
{
    /** synchronize all threads within a thread block */
    template<concepts::Acc T_Acc>
    struct Sync::Op<T_Acc, alpaka::layer::Block>
    {
        constexpr auto operator()(T_Acc const& acc, alpaka::layer::Block const scope) const
        {
            alpaka::unused(scope);
            acc[action::threadBlockSync]();
        }
    };
} // namespace alpaka::onAcc::internalCompute
