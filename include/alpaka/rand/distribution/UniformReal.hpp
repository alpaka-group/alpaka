/* Copyright 2025 Mehmet Yusufoglu, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/concepts.hpp"
#include "alpaka/rand/distribution/interval.hpp"
#include "alpaka/rand/engine/philox/philox.hpp"

namespace alpaka::rand::distribution::internal
{
    /** Returns a constant, which is equivalent to std::nextafter(T_Floating{1}, T_Floating{0})
     * or more specifically: returns the highest floating-point value lower than one.
     * There does not seem to exist a representation of this particular floating-point number in std::numerical_limits
     * and std::nextafter is not constexpr(cpp20).
     */
    template<std::floating_point T_Floating>
    consteval T_Floating prevOne() noexcept
    {
        if constexpr(sizeof(T_Floating) == 4)
        {
            return std::bit_cast<T_Floating>(static_cast<uint32_t>(0x3f7f'ffff));
        }
        else if constexpr(sizeof(T_Floating) == 8)
        {
            return std::bit_cast<T_Floating>(static_cast<uint64_t>(0x3fef'ffff'ffff'ffff));
        }
    }

    /**
     * Contains some (constexpr-)constants for random bit integer to floating point conversion -- which
     * improve readability.
     */
    template<std::unsigned_integral T_Integer, std::floating_point T_Floating>
    struct Constants
    {
        /// represents one bucket when converting an integer numbers to a floating point type in the range [0,1]
        static constexpr T_Floating normalizedIntervalBin
            = T_Floating{1} / static_cast<T_Floating>(std::numeric_limits<T_Integer>::max());
        /// this expression has been used by nvidia curand to respect the lower bounds criteria  -> it essentially
        /// shifts the distribution to the bin center (on the upper bounds this shift is mitigated due to rounding --
        /// meaning 1 is not exceeded)
        static constexpr T_Floating halfBinWidth = normalizedIntervalBin / T_Floating{2};
        /// uses a slightly smaller bucket size [0,std::nextafter(1,0)] / MAX to enforce the (open-) upper bounds
        /// criteria
        static constexpr T_Floating normalizedOpenIntervalBin
            = prevOne<T_Floating>() / static_cast<T_Floating>(std::numeric_limits<T_Integer>::max());
    };

    /** Convert an integer RNG result to a floating-point value.
     *
     * This is the fallback implementation used when no interval specialization
     * matches. It should never be instantiated and exists only to
     * catch unsupported interval configurations.
     */
    template<typename T_Engine, concepts::Interval T_Interval, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion;

    /** Converts an integer RNG output to a floating-point type in the interval [0, 1).
     * The value is mapped to an interval in the range [0,std::nextafter(1,0)),
     * where std::nextafter(1,0) represents the highest representable floating-point value lower than one.
     */
    template<typename T_Engine, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::CO, T_Integer, T_Floating>
    {
        constexpr auto operator()(T_Integer const& i) const
        {
            constexpr auto interval = Constants<T_Integer, T_Floating>::normalizedOpenIntervalBin;
            return static_cast<T_Floating>(i) * interval;
        };
    };

    /** Convert an integer RNG output to a floating-point type in the interval (0, 1).
     *  The value is mapped to an interval in the range [0,std::nextafter(1,0)),
     * where std::nextafter(1,0) represents the highest representable floating-point value lower than one.
     * Afterward an offset is added to avoid generating zero - inspired by Nvidias curand approach.
     * **Note:** The bounds criteria is only strictly valid for the base interval (0, 1).
     * When applying a scaling factor > 1, rounding effects may still cause the lower or upper bound
     * to be hit. See **scaleInterval()** for the post-scaling correction.
     */
    template<typename T_Engine, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::OO, T_Integer, T_Floating>
    {
        constexpr auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) * Constants<T_Integer, T_Floating>::normalizedOpenIntervalBin
                   + Constants<T_Integer, T_Floating>::halfBinWidth;
        };
    };

    /** Convert an integer RNG output to a floating-point type in the interval (0, 1].
     * Afterward an offset is added to avoid generating zero - inspired by Nvidias curand approach.
     * **Note:** The bounds criteria is only strictly valid for the base interval (0, 1).
     * When applying a scaling factor > 1, rounding effects may still cause the lower or upper bound
     * to be hit. See **scaleInterval()** for the post-scaling correction.
     */
    template<typename T_Engine, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::OC, T_Integer, T_Floating>
    {
        constexpr auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) * Constants<T_Integer, T_Floating>::normalizedIntervalBin
                   + Constants<T_Integer, T_Floating>::halfBinWidth;
        };
    };

    /** Converts an integer RNG output to a floating point type in the closed interval [0, 1].
     */
    template<typename T_Engine, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::CC, T_Integer, T_Floating>
    {
        constexpr auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) * Constants<T_Integer, T_Floating>::normalizedIntervalBin;
        };
    };

    /** Adapt the bit length of the engine output to match the target type.
     * This is the default case where the engine result type already matches and thus the engine is simply invoked.
     */
    template<typename T_Engine, uint32_t T_byteLengthEngineResult, uint32_t T_byteLengthRealType>
    struct BitLengthConformityAdapter
    {
        static_assert(
            (T_byteLengthEngineResult == 4u || T_byteLengthRealType == 8u),
            "Result returned by the randomBitGenerator does not have a length that is accepted by the uniformReal "
            "distribution!");
        static_assert(
            (T_byteLengthEngineResult == 8u || T_byteLengthRealType == 4u),
            "The requested floating point type does not have a length that is accepted by the uniformReal "
            "distribution!");
        static_assert(
            T_byteLengthEngineResult == T_byteLengthRealType,
            "By logic this should never fail in case the compiler accepts the specialization of the adapter!");

        constexpr auto operator()(T_Engine& engine)
        {
            return engine();
        }
    };

    /** Adapts a 32-bit engine output to a 64-bit value. This involves invoking the engine twice. */
    template<typename T_Engine>
    struct BitLengthConformityAdapter<T_Engine, 4u, 8u>
    {
        constexpr auto operator()(T_Engine& engine)
        {
            return static_cast<uint64_t>(engine()) << 32u | static_cast<uint64_t>(engine());
        }
    };

    /** Adapt a 64-bit engine output to a 32-bit value. Uses a simple narrowing conversion.*/
    template<typename T_Engine>
    struct BitLengthConformityAdapter<T_Engine, 8u, 4u>
    {
        constexpr auto operator()(T_Engine& engine)
        {
            return static_cast<uint32_t>(engine());
        }
    };

    /** Generate a floating-point value in the requested interval.
     *
     * Adapts the engine output to the required bit length and converts the integer
     * to a normalized floating point value in the requested interval */
    template<concepts::Interval T_Interval, std::floating_point T_Result, typename T_Engine>
    constexpr auto getNormalizedUniformReal(T_Engine& engine) -> T_Result
    {
        using EngineResult = std::remove_cvref_t<decltype(engine())>;
        // generates an integer the length of the size T_Result
        auto adaptedBits = BitLengthConformityAdapter<
            T_Engine,
            static_cast<uint32_t>(sizeof(EngineResult)),
            static_cast<uint32_t>(sizeof(T_Result))>{}(engine);
        // convert randomBits into the required floating-point type, while respecting the requested bounds criteria
        return IntervalAwareConversion<T_Engine, T_Interval, ALPAKA_TYPEOF(adaptedBits), T_Result>{}(adaptedBits);
    }
    template<concepts::UniformVectorEngine T_Engine, uint32_t T_resultSize, uint32_t T_elemSize, uint32_t T_elems>
    struct vectorDispatchWrapper;

    template<concepts::UniformVectorEngine T_Engine, uint32_t T_elemSize, uint32_t T_elems>
    struct vectorDispatchWrapper<T_Engine, 4u, T_elemSize, T_elems>
    {
        T_Engine& ph;
        static_assert(T_elems > 0, "RandomEngine did not return any elements!");

        constexpr explicit vectorDispatchWrapper(T_Engine& eng) : ph(eng)
        {
        }

        constexpr uint32_t operator()() const
        {
            auto res = ph();
            return static_cast<uint32_t>(res[0]);
        }
    };

    /// **Wrapper specialization enabling efficient generation of 64-bit values from vectorized engines without
    /// requiring two engine calls.**
    template<concepts::UniformVectorEngine T_Engine, uint32_t T_elems>
    struct vectorDispatchWrapper<T_Engine, 8u, 4u, T_elems>
    {
        T_Engine& ph;
        using TResult = decltype(ph());
        static constexpr auto dim = TResult::dim();
        static_assert(T_elems >= 2, "Engine result dimension must be >= 2, to be usable in UniformReal<double>");

        constexpr explicit vectorDispatchWrapper(T_Engine& eng) : ph(eng)
        {
        }

        constexpr uint64_t operator()() const
        {
            auto res = ph();
            return (static_cast<uint64_t>(res[0]) << 32u) | static_cast<uint64_t>(res[1]);
        }
    };

    template<std::floating_point T_Floating, concepts::Interval T_Interval>
    class UniformRealBase
    {
    public:
        using ResultType = T_Floating;

        using IntervalType = T_Interval;

        constexpr explicit UniformRealBase(T_Floating min, T_Floating max, [[maybe_unused]] T_Interval)
            : m_min(min)
            , m_max(max)
            , m_range(m_max - m_min) // abs is a fail-safe in case min>max
        {
        }

    protected:
        T_Floating const m_min;
        T_Floating const m_max;
        T_Floating const m_range;
    };
} // namespace alpaka::rand::distribution::internal

namespace alpaka::rand::distribution
{
    /** Select a floating-point value from a uniform interval.
     *
     * This generator produces floating-point values of type `T_Result` drawn from a uniform
     * interval `[a, b)` or `(a, b]`, depending on the interval type specified via `Interval_v`: default case is CO
     * ->[a,b). The interface mirrors `std::uniform_real_distribution`, and can be invoked with any engine
     * adhering to the 'UniformRandomEngine' concept, which includes std uniform engines.
     *
     * **Supported result types:** `float`, `double`
     * **Supported engine result widths:** 32-bit and 64-bit unsigned integers.
     *
     * @note This distribution is subject to a small non-uniform bias; see
     *       **UniformReal::operator()** for details.
     */
    template<std::floating_point T_Result, concepts::Interval T_Interval = interval::CO>
    struct UniformReal : internal::UniformRealBase<T_Result, T_Interval>
    {
        static_assert(static_cast<uint32_t>(sizeof(T_Result)) == 4u || static_cast<uint32_t>(sizeof(T_Result)) == 8u);

        template<std::integral T_Value>
        static consteval void checkValueConformity()
        {
            static_assert(
                static_cast<uint32_t>(sizeof(T_Value)) == 4u || static_cast<uint32_t>(sizeof(T_Value)) == 8u);
        }

        using Base = internal::UniformRealBase<T_Result, T_Interval>;

        constexpr explicit UniformReal([[maybe_unused]] T_Interval interval = T_Interval{}) : Base(0, 1, interval)
        {
        }

        constexpr UniformReal(T_Result min, T_Result max, [[maybe_unused]] T_Interval interval = T_Interval{})
            : Base(min, max, interval)
        {
        }

        /** Selects a value from a uniform distribution over the configured (min, max) interval,
         * respecting the specified interval bounds.**
         *
         * **Input:** a random engine conforming to the `UniformRandomEngine` concept
         * (currently accepts stdlib uniform engines and alpaka engines included in the alpaka::rand::engine namespace)
         * **Output:** a floating-point value sampled from the configured distribution.
         *
         * @note: This distribution introduces a slight numerical bias due to floating-point
         * rounding effects and the use of a **1 / MAX** integer-to-floating-point conversion methods
         * @see Goualard, F. (2020). Generating Random Floating-Point Numbers by Dividing Integers: A Case Study.
         * https://doi.org/10.1007/978-3-030-50417-5_2 or Allen B. Downey Generating Pseudo-random Floating-Point
         * Values https://allendowney.com/research/rand/.
         * Additionally, using an interval with an open bound
         * (OC,CO,OO) may introduce yet another small non-uniform bias -- @see scaleInterval().
         */
        template<concepts::UniformRandomEngine T_Engine>
        constexpr auto operator()(T_Engine& engine) const -> T_Result
        {
            return engineDispatch(engine);
        }

    private:
        /** Dispatch for std engines and alpaka abbreviations conforming to the std::uniform_random_bit_generator
         * concept (e.g. Philox4x32x10)
         */
        template<concepts::UniformStdEngine T_Engine>
        constexpr auto engineDispatch(T_Engine& engine) const -> T_Result
        {
            using EngineResult = ALPAKA_TYPEOF(engine());
            checkValueConformity<EngineResult>();
            T_Result res = internal::getNormalizedUniformReal<T_Interval, T_Result, T_Engine>(engine);
            // @TODO potentially add underflow protection as suggested by https://doi.org/10.1145/3503512
            return scaleInterval(res);
        }

        /** Dispatch for vector engines (uniform bit generators that return a vector (e.g Philox4x32x10Vector) to
         * enable efficient double precision uniform_real generation (reducing the number of invocations)
         */
        template<concepts::UniformVectorEngine T_Engine>
        constexpr auto engineDispatch(T_Engine& engine) const -> T_Result
        {
            using EngineResult = ALPAKA_TYPEOF(engine());
            using ValueType = typename EngineResult::value_type;
            checkValueConformity<ValueType>();
            constexpr auto dim = getDim(EngineResult{});
            auto dispatchWrapper = internal::vectorDispatchWrapper<
                T_Engine,
                static_cast<uint32_t>(sizeof(T_Result)),
                static_cast<uint32_t>(sizeof(ValueType)),
                dim>(engine);
            using DispatchWrapper = decltype(dispatchWrapper);
            T_Result res = internal::getNormalizedUniformReal<T_Interval, T_Result, DispatchWrapper>(dispatchWrapper);
            return scaleInterval(res);
        }

        /** For open bounds, the normalized value in (0, 1) may (still) hit the lower or upper bound
         * due to floating-point rounding (after scaling is applied).
         *
         * To enforce adherence to the requested interval, the result is shifted to the next representable
         * value using std::nextafter.
         * WARNING: std::nextafter is not constexpr in cpp20 -> this might cause problems on some devices (regarding
         * missing __device__ annotations) and requires often unnecessary runtime evaluation -- future
         * maintainers should consider creating a constexpr adaptation of std::nextafter
         *
         * @note This introduces a(-/another) small non-uniform bias. The current implementation is inherently
         *       non-uniform due to integer-to-floating-point mapping @see  https://doi.org/10.1007/978-3-030-50417-5_2


         */
        constexpr auto scaleInterval(T_Result const& normalizedVal) const -> T_Result
        {
            T_Result res = normalizedVal * this->m_range + this->m_min;

            if constexpr(std::is_same_v<T_Interval, interval::OC> || std::is_same_v<T_Interval, interval::OO>)
            {
                if(res == this->m_min)
                    res = std::nextafter(this->m_min, this->m_max);
            }
            if constexpr(std::is_same_v<T_Interval, interval::CO> || std::is_same_v<T_Interval, interval::OO>)
            {
                if(res == this->m_max)
                    res = std::nextafter(this->m_max, this->m_min);
            }
            return res;
        }
    };
} // namespace alpaka::rand::distribution
