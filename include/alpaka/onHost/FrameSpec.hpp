/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/executor.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka::onHost
{
    /** @brief Device/Api-agnostic description of the logical parallelism exposed to a kernel.
     *
     * A frame specification describes how a multidimensional index range [0; K) is divided into fixed-size chunks,
     * called frames (NF), each with a frame extent (FE), where `K = NF * FE`.
     * K does not need to match the problem size (P), e.g., the number of elements in a buffer you want to process in a
     * kernel. Often, the best performance of a kernel can be achieved if `K <= P`, and if the
     * kernel uses SIMD operations, `K <= P/(SIMD width)`.
     * alpaka derives the onHost::ThreadSpec to launch the kernel, based on a heuristic and additional launch
     * information from the `FrameSpec`. Therefore a kernel enqueued with a frame specification should always be
     * written to be executable with any onHost::ThreadSpec and should not depend on hard-coded thread numbers, to
     * ensure portability between devices.
     *
     * A `FrameSpec` is therefore not equivalent to a CUDA-style grid description. It specifies only the maximum
     * parallelism made available to the kernel. It does not guarantee the number of physical thread blocks, nor the
     * number of physical threads per block used by the backend. If exact control over blocks and threads is required,
     * use onHost::ThreadSpec.
     *
     * @tparam T_NumFrames The n-dimensional number of frames.
     * @tparam T_FrameExtents The n-dimensional size of one logical frame.
     * @tparam T_Executor The executor used to translate the onHost::ThreadSpec into a thread block hierarchy.
     * If the executor is exec::AnyExecutor alpaka will select a good fitting executor for the action where the
     * ThreadSpec is used.
     */
    template<
        alpaka::concepts::Vector T_NumFrames,
        alpaka::concepts::Vector<typename T_NumFrames::value_type, T_NumFrames::dim()> T_FrameExtents,
        alpaka::concepts::Executor T_Executor = alpaka::exec::AnyExecutor>
    struct FrameSpec
    {
        using index_type = typename T_NumFrames::value_type;

        using NumFramesVecType = T_NumFrames;
        using FrameExtentsVecType = T_FrameExtents;

    private:
        NumFramesVecType m_numFrames;
        FrameExtentsVecType m_frameExtents;

    public:
        constexpr FrameSpec(
            T_NumFrames const& numFrames,
            T_FrameExtents const& frameExtent,
            T_Executor executor = T_Executor{})
            : m_numFrames(numFrames)
            , m_frameExtents(frameExtent)
        {
            alpaka::unused(executor);
        }

        [[nodiscard]] static constexpr T_Executor getExecutor() noexcept
        {
            return T_Executor{};
        }

        [[nodiscard]] constexpr NumFramesVecType const& getNumFrames() const noexcept
        {
            return m_numFrames;
        }

        [[nodiscard]] constexpr FrameExtentsVecType const& getFrameExtents() const noexcept
        {
            return m_frameExtents;
        }

        [[nodiscard]] static consteval uint32_t dim()
        {
            return T_FrameExtents::dim();
        }
    };

    template<alpaka::concepts::VectorOrScalar T_NumFrames, alpaka::concepts::VectorOrScalar T_FrameExtents>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&) -> FrameSpec<
        alpaka::trait::GetVec_t<T_NumFrames>,
        alpaka::trait::GetVec_t<T_FrameExtents>,
        alpaka::exec::AnyExecutor>;

    template<
        alpaka::concepts::VectorOrScalar T_NumFrames,
        alpaka::concepts::VectorOrScalar T_FrameExtents,
        alpaka::concepts::Executor T_Executor>
    FrameSpec(T_NumFrames const&, T_FrameExtents const&, T_Executor)
        -> FrameSpec<alpaka::trait::GetVec_t<T_NumFrames>, alpaka::trait::GetVec_t<T_FrameExtents>, T_Executor>;

    namespace trait
    {
        template<typename T>
        struct IsFrameSpec : std::false_type
        {
        };

        template<
            alpaka::concepts::Vector T_NumFrames,
            alpaka::concepts::Vector T_FrameExtents,
            alpaka::concepts::Executor T_Executor>
        struct IsFrameSpec<onHost::FrameSpec<T_NumFrames, T_FrameExtents, T_Executor>> : std::true_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isFrameSpec_v = trait::IsFrameSpec<T>::value;

    namespace concepts
    {
        /** Concept to check if a type is a FrameSpec
         *
         * @tparam T Type to check
         * @tparam T_IndexType enforce an index type of the frame specification, if not provided the type is not
         * checked
         * @tparam T_dim enforce a dimensionality of the frame specification, if not provided the value is not
         * checked
         */
        template<typename T, typename T_IndexType = alpaka::NotRequired, uint32_t T_dim = alpaka::notRequiredDim>
        concept FrameSpec
            = isFrameSpec_v<T>
              && (std::same_as<T_IndexType, alpaka::NotRequired> || std::same_as<typename T::index_type, T_IndexType>)
              && ((T_dim == alpaka::notRequiredDim) || (T::dim() == T_dim));

        /** Concept to check if a type is a ThreadSpec or a FrameSpec
         *
         * @tparam T Type to check
         */
        template<typename T>
        concept ThreadOrFrameSpec = isFrameSpec_v<T> || isThreadSpec_v<T>;
    } // namespace concepts

    std::ostream& operator<<(std::ostream& s, concepts::FrameSpec auto const& d)
    {
        return s << "FrameSpec{ frames=" << d.getNumFrames() << ", frameExtent=" << d.getFrameExtents() << " }";
    }

} // namespace alpaka::onHost
