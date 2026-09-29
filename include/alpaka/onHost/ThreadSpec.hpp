/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/executor.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka::onHost
{
    /** @brief Backend-specific description of the actual block and thread launch shape.
     *
     * A thread specification directly describes the number of blocks and the number of threads per block that are
     * passed to the backend. This is the closest alpaka equivalent to a CUDA grid/block launch configuration.
     *
     * In contrast to `alpaka::onHost::FrameSpec`, a `ThreadSpec` is not a logical frame decomposition. It is used
     * when a kernel requires exact guarantees about the number of blocks and the size of each block.
     *
     * @tparam T_Executor If the executor is alpaka::exec::AnyExecutor alpaka will select a good fitting executor for
     * the action where the ThreadSpec is used.
     */
    template<
        alpaka::concepts::Vector T_NumBlocks,
        alpaka::concepts::Vector<typename T_NumBlocks::value_type, T_NumBlocks::dim()> T_NumThreads,
        alpaka::concepts::Executor T_Executor = alpaka::exec::AnyExecutor>
    struct ThreadSpec
    {
        using index_type = typename T_NumBlocks::value_type;
        using NumBlocksVecType = typename T_NumBlocks::UniVec;
        using NumThreadsVecType = T_NumThreads;

    private:
        NumBlocksVecType m_numBlocks;
        NumThreadsVecType m_numThreads;

    public:
        constexpr ThreadSpec(
            T_NumBlocks const& numBlocks,
            T_NumThreads const& numThreadsPerBlock,
            T_Executor executor = T_Executor{})
            : m_numBlocks(numBlocks)
            , m_numThreads(numThreadsPerBlock)
        {
            alpaka::unused(executor);
        }

        [[nodiscard]] static constexpr T_Executor getExecutor()
        {
            return T_Executor{};
        }

        [[nodiscard]] constexpr NumThreadsVecType const& getNumThreads() const noexcept
        {
            return m_numThreads;
        }

        [[nodiscard]] constexpr NumBlocksVecType const& getNumBlocks() const noexcept
        {
            return m_numBlocks;
        }

        [[nodiscard]] static consteval uint32_t dim()
        {
            return T_NumThreads::dim();
        }
    };

    template<alpaka::concepts::VectorOrScalar T_NumBlocks, alpaka::concepts::VectorOrScalar T_NumThreads>
    ThreadSpec(T_NumBlocks const&, T_NumThreads const&)
        -> ThreadSpec<alpaka::trait::GetVec_t<T_NumBlocks>, alpaka::trait::GetVec_t<T_NumThreads>>;

    template<
        alpaka::concepts::VectorOrScalar T_NumBlocks,
        alpaka::concepts::VectorOrScalar T_NumThreads,
        alpaka::concepts::Executor T_Executor>
    ThreadSpec(T_NumBlocks const&, T_NumThreads const&, T_Executor)
        -> ThreadSpec<alpaka::trait::GetVec_t<T_NumBlocks>, alpaka::trait::GetVec_t<T_NumThreads>, T_Executor>;

    namespace trait
    {
        template<typename T>
        struct IsThreadSpec : std::false_type
        {
        };

        template<
            alpaka::concepts::Vector T_NumBlocks,
            alpaka::concepts::Vector T_NumThreads,
            alpaka::concepts::Executor T_Executor>
        struct IsThreadSpec<onHost::ThreadSpec<T_NumBlocks, T_NumThreads, T_Executor>> : std::true_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isThreadSpec_v = trait::IsThreadSpec<T>::value;

    namespace concepts
    {
        /** Concept to check if a type is a ThreadSpec
         *
         * @tparam T Type to check
         * @tparam T_IndexType enforce an index type of the thread specification, if not provided the type is not
         * checked
         * @tparam T_dim enforce a dimensionality of the thread specification, if not provided the value is not
         * checked
         */
        template<typename T, typename T_IndexType = alpaka::NotRequired, uint32_t T_dim = alpaka::notRequiredDim>
        concept ThreadSpec
            = isThreadSpec_v<T>
              && (std::same_as<T_IndexType, alpaka::NotRequired> || std::same_as<typename T::index_type, T_IndexType>)
              && ((T_dim == alpaka::notRequiredDim) || (T::dim() == T_dim));
    } // namespace concepts

    std::ostream& operator<<(std::ostream& s, concepts::ThreadSpec auto const& t)
    {
        return s << "ThreadSpec{ blocks=" << t.getNumBlocks() << ", threads=" << t.getNumThreads() << " }";
    }
} // namespace alpaka::onHost
