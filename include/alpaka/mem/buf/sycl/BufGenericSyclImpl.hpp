/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka::detail
{
    //! The Sycl memory buffer implementation.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class BufGenericSyclImpl final
    {
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

    public:
        template<typename TExtent>
        ALPAKA_FN_HOST BufGenericSyclImpl(
            DevGenericSycl<TTag> dev,
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

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem) << std::endl;
#    endif
        }

        BufGenericSyclImpl(BufGenericSyclImpl&&) = delete;
        auto operator=(BufGenericSyclImpl&&) -> BufGenericSyclImpl& = delete;

        ALPAKA_FN_HOST ~BufGenericSyclImpl()
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // NOTE: m_pMem is allowed to be a nullptr here.
            m_deleter(m_pMem);
        }

    private:
        DevGenericSycl<TTag> m_dev;
        TElem* const m_pMem;
        std::function<void(TElem*)> m_deleter;
        Vec<TDim, TIdx> m_extentElements;

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

#endif
