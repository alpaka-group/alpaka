/* Copyright 2022 Jiří Vyskočil, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/meta/IsArrayOrVector.hpp"
#include "alpaka/rand/Philox/PhiloxSingle.hpp"
#include "alpaka/rand/Philox/PhiloxVector.hpp"
#include "alpaka/rand/Traits.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace alpaka::rand
{
    /** Most common Philox engine variant, outputs single number
     *
     * This is a variant of the Philox engine generator which outputs a single float. The counter size is \f$4
     * \times 32 = 128\f$ bits. Since the engine returns a single number, the generated result, which has the same
     * size as the counter, has to be stored between invocations. Additionally a 32 bit pointer is stored. The
     * total size of the state is 352 bits = 44 bytes.
     *
     * Ref.: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and
     * Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     */
    class Philox4x32x10 : public interface::Implements<ConceptRand, Philox4x32x10>
    {
    public:
        /// Philox algorithm: 10 rounds, 4 numbers of size 32.
        using EngineParams = engine::PhiloxParams<4, 32, 10>;
        /// Engine outputs a single number
        using EngineVariant = engine::PhiloxSingle<EngineParams>;

        /** Initialize a new Philox engine
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset numbers from the start of the subsequence
         */
        ALPAKA_FN_HOST_ACC Philox4x32x10(
            std::uint64_t const seed = 0,
            std::uint64_t const subsequence = 0,
            std::uint64_t const offset = 0)
            : engineVariant(seed, subsequence, offset)
        {
        }

        // STL UniformRandomBitGenerator concept
        // https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator
        using result_type = std::uint32_t;

        ALPAKA_FN_HOST_ACC constexpr auto min() -> result_type
        {
            return 0;
        }

        ALPAKA_FN_HOST_ACC constexpr auto max() -> result_type
        {
            return std::numeric_limits<result_type>::max();
        }

        ALPAKA_FN_HOST_ACC auto operator()() -> result_type
        {
            return engineVariant();
        }

    private:
        EngineVariant engineVariant;
    };

    /** Most common Philox engine variant, outputs a 4-vector of floats
     *
     * This is a variant of the Philox engine generator which outputs a vector containing 4 floats. The counter
     * size is \f$4 \times 32 = 128\f$ bits. Since the engine returns the whole generated vector, it is up to the
     * user to extract individual floats as they need. The benefit is smaller state size since the state does not
     * contain the intermediate results. The total size of the state is 192 bits = 24 bytes.
     *
     * Ref.: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and
     * Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     */
    class Philox4x32x10Vector : public interface::Implements<ConceptRand, Philox4x32x10Vector>
    {
    public:
        using EngineParams = engine::PhiloxParams<4, 32, 10>;
        using EngineVariant = engine::PhiloxVector<EngineParams>;

        /** Initialize a new Philox engine
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Number of vectors to skip from the start of the subsequence. Vector i is the same as
         *        numbers 4 (offset + i) to 4 (offset + i) + 3 of Philox4x32x10 with the same seed and subsequence.
         */
        ALPAKA_FN_HOST_ACC Philox4x32x10Vector(
            std::uint64_t const seed = 0,
            std::uint64_t const subsequence = 0,
            std::uint64_t const offset = 0)
            : engineVariant(seed, subsequence, offset)
        {
        }

        template<typename TScalar>
        using ResultContainer = typename EngineVariant::template ResultContainer<TScalar>;

        using ResultInt = std::uint32_t;
        using ResultVec = decltype(std::declval<EngineVariant>()());

        ALPAKA_FN_HOST_ACC constexpr auto min() -> ResultInt
        {
            return 0;
        }

        ALPAKA_FN_HOST_ACC constexpr auto max() -> ResultInt
        {
            return std::numeric_limits<ResultInt>::max();
        }

        ALPAKA_FN_HOST_ACC auto operator()() -> ResultVec
        {
            return engineVariant();
        }

    private:
        EngineVariant engineVariant;
    };

    // The following exists because you "cannot call __device__ function from a __host__ __device__ function"
    // directly, but wrapping that call in a struct is just fine.
    template<typename TEngine>
    struct EngineCallHostAccProxy
    {
        ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> decltype(engine())
        {
            return engine();
        }
    };

    /// TEMP: Distributions to be decided on later. The generator should be compatible with STL as of now.
    //! Uniform floating-point numbers in [min, max), for min < max.
    template<typename TResult, typename TSfinae = void>
    class UniformReal : public interface::Implements<ConceptRand, UniformReal<TResult>>
    {
        template<typename TRes, typename TEnable = void>
        struct ResultType
        {
            using type = TRes;
        };

        template<typename TRes>
        struct ResultType<TRes, std::enable_if_t<meta::IsArrayOrVector<TRes>::value>>
        {
            using type = typename TRes::value_type;
        };

        using T = typename ResultType<TResult>::type;
        static_assert(std::is_floating_point_v<T>, "Only floating-point types are supported");

    public:
        ALPAKA_FN_HOST_ACC UniformReal() : UniformReal(0, 1)
        {
        }

        ALPAKA_FN_HOST_ACC UniformReal(T min, T max)
            : _min(min)
            , _max(max)
            , _range(_max - _min)
            , _belowMax(largestBelow(max))
            , _clampNeeded(!(min == T{0} && max == T{1}))
        {
        }

        template<typename TEngine>
        ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> TResult
        {
            using Value = std::remove_cv_t<std::remove_reference_t<decltype(engine.max())>>;
            static_assert(std::is_unsigned_v<Value>, "The engine must return unsigned integers.");
            Value const engineMin = engine.min();
            Value const engineMax = engine.max();
            if constexpr(meta::IsArrayOrVector<TResult>::value)
            {
                auto result = engine();
                TResult ret{
                    scale(toUnitInterval<Value>(result[0], engineMin, engineMax)),
                    scale(toUnitInterval<Value>(result[1], engineMin, engineMax)),
                    scale(toUnitInterval<Value>(result[2], engineMin, engineMax)),
                    scale(toUnitInterval<Value>(result[3], engineMin, engineMax))};
                return ret;
            }
            else
            {
                // Since it's possible to get a host-only engine here, the call has to go through proxy
                return scale(toUnitInterval<Value>(EngineCallHostAccProxy<TEngine>{}(engine), engineMin, engineMax));
            }

            ALPAKA_UNREACHABLE(TResult{});
        }

    private:
        //! 2^-n as a T.
        static ALPAKA_FN_HOST_ACC constexpr auto inversePowerOfTwo(int n) -> T
        {
            T result = 1;
            for(; n > 0; --n)
            {
                result *= static_cast<T>(0.5);
            }
            return result;
        }

        //! Maps an engine number in [engineMin, engineMax] to [0, 1).
        //! The conversion is exact for an engine that returns all values of TValue.
        template<typename TValue>
        static ALPAKA_FN_HOST_ACC auto toUnitInterval(
            TValue const value,
            TValue const engineMin,
            TValue const engineMax) -> T
        {
            constexpr int valueBits = std::numeric_limits<TValue>::digits;
            constexpr int digits = std::numeric_limits<T>::digits;
            if(engineMin == 0 && engineMax == std::numeric_limits<TValue>::max())
            {
                if constexpr(digits < valueBits)
                {
                    return static_cast<T>(value >> (valueBits - digits)) * inversePowerOfTwo(digits);
                }
                else
                {
                    return static_cast<T>(value) * inversePowerOfTwo(valueBits);
                }
            }
            T const unit = static_cast<T>(value - engineMin) / (static_cast<T>(engineMax - engineMin) + T{1});
            constexpr T belowOne = T{1} - std::numeric_limits<T>::epsilon() / 2;
            return unit < T{1} ? unit : belowOne;
        }

        //! The largest value of T below the finite value x.
        static ALPAKA_FN_HOST_ACC auto largestBelow(T const x) -> T
        {
            if(x == std::numeric_limits<T>::infinity() || x == -std::numeric_limits<T>::infinity())
            {
                return x;
            }
            if(!(x < T{0}) && !(x > T{0}))
            {
                return -std::numeric_limits<T>::denorm_min();
            }
            if constexpr(sizeof(T) == sizeof(std::uint32_t) || sizeof(T) == sizeof(std::uint64_t))
            {
                // The integer representation steps to the next value, without a library call.
                using Bits = std::conditional_t<sizeof(T) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;
                Bits const bits = std::bit_cast<Bits>(x);
                return std::bit_cast<T>(x > T{0} ? static_cast<Bits>(bits - 1) : static_cast<Bits>(bits + 1));
            }
            else
            {
                // Long double on x86 has padding bytes instead of a plain integer representation.
                // Any value below x satisfies the upper bound, it does not have to be the
                // immediate predecessor.
                T const step = (x > T{0} ? x : -x) * std::numeric_limits<T>::epsilon();
                if(step > T{0})
                {
                    return x - step;
                }
                return x > T{0} ? T{0} : x * T{2};
            }
        }

        //! Maps u in [0, 1) to [min, max).
        ALPAKA_FN_HOST_ACC auto scale(T const unit) const -> T
        {
            // The clamp is only needed outside of [0, 1): unit is below one by construction and
            // unit * 1 + 0 is exact. The decision is computed once in the constructor, so the
            // compare does not stay in the draw loop when the distribution is a function argument.
            if(!_clampNeeded)
            {
                return unit;
            }
            T const result = unit * _range + _min;
            return result < _max ? result : _belowMax;
        }

        T const _min;
        T const _max;
        T const _range;
        T const _belowMax;
        bool const _clampNeeded;
    };
} // namespace alpaka::rand
