/* Copyright 2022 Jiří Vyskočil, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/rand/engine/philox/PhiloxSingle.hpp"
#include "alpaka/rand/engine/philox/PhiloxVector.hpp"

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace alpaka::rand::engine
{

    /** Most common Philox engine variant, outputs single number
     *
     * This is a variant of the Philox engine generator which outputs a single float. The counter size is \f$4
     * \times 32 = 128\f$ bits. A bit shuffle is performed 10 subsequent times.
     *  Since the engine returns a single number, the generated result, which has the same
     * size as the counter, has to be stored between invocations. Additionally a 32 bit pointer is stored. The
     * total size of the state is 352 bits = 44 bytes.
     *
     * Ref.: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and
     * Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     */
    class Philox4x32x10
    {
    public:
        /// Philox algorithm: 10 rounds, 4 numbers of size 32.
        using EngineParams = internal::PhiloxParams<4, 32, 10>;
        /// Engine outputs a single number
        using EngineVariant = internal::PhiloxSingle<EngineParams>;

        /** Initialize a new Philox engine
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset numbers from the start of the subsequence
         */
        constexpr explicit Philox4x32x10(
            std::uint64_t const seed = 0,
            std::uint64_t const subsequence = 0,
            std::uint64_t const offset = 0)
            : m_engineVariant(seed, subsequence, offset)
        {
        }

        // STL UniformRandomBitGenerator concept
        // See the functions min and max for the range of the generated numbers
        // https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator
        using ResultType = std::uint32_t;

        static constexpr auto min() -> ResultType
        {
            return 0;
        }

        static constexpr auto max() -> ResultType
        {
            return std::numeric_limits<ResultType>::max();
        }

        constexpr auto operator()() -> ResultType
        {
            return m_engineVariant();
        }

    private:
        EngineVariant m_engineVariant;
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
    class Philox4x32x10Vector
    {
    public:
        using EngineParams = internal::PhiloxParams<4, 32, 10>;
        using EngineVariant = internal::PhiloxVector<EngineParams>;

        /** Initialize a new Philox engine
         *
         * Vector i at offset n is the same as numbers 4 (n + i) to 4 (n + i) + 3 of Philox4x32x10 with
         * the same seed and subsequence.
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Number of vectors to skip from the start of the subsequence.
         */
        constexpr explicit Philox4x32x10Vector(
            std::uint64_t const seed = 0,
            std::uint64_t const subsequence = 0,
            std::uint64_t const offset = 0)
            : m_engineVariant(seed, subsequence, offset)
        {
        }

        template<typename T_Scalar>
        using ResultContainer = EngineVariant::ResultContainer<T_Scalar>;

        using ResultInt = std::uint32_t;
        using ResultVec = decltype(std::declval<EngineVariant>()());

        static constexpr auto min() -> ResultInt
        {
            return 0;
        }

        static constexpr auto max() -> ResultInt
        {
            return std::numeric_limits<ResultInt>::max();
        }

        constexpr auto operator()() -> ResultVec
        {
            return m_engineVariant();
        }

    private:
        EngineVariant m_engineVariant;
    };


} // namespace alpaka::rand::engine
