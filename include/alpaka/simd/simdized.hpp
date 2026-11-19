/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 * @brief This file provides functionality to transform a type into a SIMD-optimized data structure.
 *
 * The implementation is motivated by https://ieeexplore.ieee.org/document/11207437
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/core/common.hpp"

namespace alpaka
{

    /** Transform a type into a SIMD-optimized data structure.
     *
     * A simdized value is not necessarily a data type wrapped by alpaka::Simd, it can be a structured hierarchical
     * type where each component is a SIMD pack. This function can be specialized within the namespace of the input
     * type and will be found via ADL.
     * @see alpakaSimdizedInvoke should be specialized together with this function.
     *
     * Simdizing of structured types as shown in the code example often improves the performance compared to wrapping
     * into a SIMD pack.
     * @code{.cpp}
     * // input type
     * template<typename T>
     * struct Pos
     * {
     *   T x = 0;
     *   T y = 1;
     * };
     *
     * // output could be
     * template<typename T>
     * struct Pos
     * {
     *   alpaka::Simd<T, width> x;
     *   alpaka::Simd<T, width> y;
     * };
     * @endcode
     *
     * @tparam T_width the width of the used SIMD type
     * @return A simdized data type where each lane replicates the given value. If `makeSimdized` is not specialized
     * for the given type a SIMD pack wrapping the input value will be returned.
     */
    template<uint32_t T_width>
    constexpr auto makeSimdized(auto&& value)
    {
        return Simd<ALPAKA_TYPEOF(value), T_width>::fill(value);
    }

    /** Invokes the callable object fn with the parameters args.
     *
     * For structured data where each component is a SIMD pack, the functor should be forwarded to the members while
     * recursively calling alpakaSimdizedInvoke.
     * As soon as there is no use specialization available, the recursion is terminated by the invocation of the
     * functor with the forwarded arguments. This function can be specialized within the namespace of the argument
     * types and will be found via ADL.
     * @see makeSimdized should be specialized together with this function.
     *
     * As shown in the code snippet, alpaka assumes at least a specialization where each argument can perform the same
     * access used within the function. It is allowed to specialize more function signatures that do not follow the
     * rule but are useful within the user code.
     * @code{.cpp}
     * // A typical case of how this specialization is called is
     * // `alpakaSimdizedInvoke(f, Pos<int>{}, Pos<alpaka::Simd<int,4>>{})`.
     * constexpr void alpakaSimdizedInvoke(auto&& f, alpaka::concepts::SpecializationOf<Pos> auto&&... args)
     * {
     *    // Accessing .x and .y must be supported by all arguments.
     *    alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).x...);
     *    alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).y...);
     * }
     * @endcode
     *
     * @param fn Callable object to which the arguments will be forwarded.
     * @param args Arguments forwarded to fn.
     */
    constexpr void alpakaSimdizedInvoke(auto&& fn, auto&&... args)
    {
        ALPAKA_FORWARD(fn)(ALPAKA_FORWARD(args)...);
    }
} // namespace alpaka
