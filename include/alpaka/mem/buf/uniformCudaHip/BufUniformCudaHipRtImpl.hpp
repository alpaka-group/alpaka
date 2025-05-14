/* Copyright 2025 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dev/DevCudaRt.hpp"
#include "alpaka/dev/DevHipRt.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::detail
{
    template<typename TDim, typename SFINAE = void>
    struct PitchHolder
    {
        explicit PitchHolder(std::size_t)
        {
        }
    };

    template<typename TDim>
    struct PitchHolder<TDim, std::enable_if_t<TDim::value >= 2>>
    {
        std::size_t m_rowPitchInBytes;
    };

    //! The Uniform Cuda/HIP memory buffer implementation.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRtImpl final : detail::PitchHolder<TDim>
    {
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

    public:
        template<typename TExtent>
        ALPAKA_FN_HOST BufUniformCudaHipRtImpl(
            DevUniformCudaHipRt<TApi> dev,
            TElem* pMem,
            std::function<void(TElem*)> deleter,
            TExtent const& extent,
            std::size_t pitchBytes)
            : detail::PitchHolder<TDim>{pitchBytes}
            , m_dev(dev)
            , m_pMem(pMem)
            , m_deleter(std::move(deleter))
            , m_extentElements(getExtents(extent))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");
            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem) << std::endl;
#    endif
        }

        BufUniformCudaHipRtImpl(BufUniformCudaHipRtImpl&&) = delete;
        auto operator=(BufUniformCudaHipRtImpl&&) -> BufUniformCudaHipRtImpl& = delete;

        ALPAKA_FN_HOST ~BufUniformCudaHipRtImpl()
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // NOTE: m_pMem is allowed to be a nullptr here.
            m_deleter(m_pMem);
        }

    private:
        DevUniformCudaHipRt<TApi> const m_dev;
        TElem* const m_pMem;
        std::function<void(TElem*)> m_deleter;
        Vec<TDim, TIdx> const m_extentElements;

        // friend declarations to allow usage of these pointers in the respective trait implementations
        template<typename TBuf, typename TSfinae>
        friend struct alpaka::trait::GetDev;

        template<typename TBuf, typename TSfinae>
        friend struct alpaka::trait::GetExtents;

        template<typename TBuf, typename TSfinae>
        friend struct alpaka::trait::GetPtrNative;

        template<typename TBuf, typename TDev, typename TSfinae>
        friend struct alpaka::trait::GetPtrDev;

        template<typename TBuf, typename TSfinae>
        friend struct alpaka::trait::GetPitchesInBytes;
    };

} // namespace alpaka::detail

#endif
