/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>

namespace alpaka::detail
{
    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpuImpl final
    {
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

    public:
        template<typename TExtent>
        ALPAKA_FN_HOST BufCpuImpl(
            DevCpu dev,
            TElem* pMem,
            std::function<void(TElem*)> deleter,
            TExtent const& extent) noexcept
            : m_dev(std::move(dev))
            , m_pMem(pMem)
            , m_deleter(std::move(deleter))
            , m_extentElements(getExtentVecEnd<TDim>(extent))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");
            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem) << std::endl;
#endif
        }

        BufCpuImpl(BufCpuImpl&&) = delete;
        auto operator=(BufCpuImpl&&) -> BufCpuImpl& = delete;

        ALPAKA_FN_HOST ~BufCpuImpl()
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // NOTE: m_pMem is allowed to be a nullptr here.
            m_deleter(m_pMem);
        }

    private:
        DevCpu const m_dev;
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
    };
} // namespace alpaka::detail
