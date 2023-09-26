/*
 * SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-FileContributor: Luca Ferragina <luca.ferragina@cern.ch>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 * SPDX-FileContributor: Jan Stephan <j.stephan@hzdr.de>
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/shared/st/Traits.hpp"
#include "alpaka/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp"

#include <cstddef>
#include <cstdint>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The generic SYCL shared memory allocator.
    class BlockSharedMemStGenericSycl
        : public alpaka::detail::BlockSharedMemStMemberImpl<>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStGenericSycl>
    {
    public:
        BlockSharedMemStGenericSycl(sycl::local_accessor<std::byte> accessor)
            : BlockSharedMemStMemberImpl(
                reinterpret_cast<std::uint8_t*>(accessor.get_pointer().get()),
                accessor.size())
            , m_accessor{accessor}
        {
        }

    private:
        sycl::local_accessor<std::byte> m_accessor;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename T, std::size_t TUniqueId>
    struct DeclareSharedVar<T, TUniqueId, BlockSharedMemStGenericSycl>
    {
        static auto declareVar(BlockSharedMemStGenericSycl const& smem) -> T&
        {
            auto* data = smem.template getVarPtr<T>(TUniqueId);

            if(!data)
            {
                smem.template alloc<T>(TUniqueId);
                data = smem.template getLatestVarPtr<T>();
            }
            ALPAKA_ASSERT(data != nullptr);
            return *data;
        }
    };

    template<>
    struct FreeSharedVars<BlockSharedMemStGenericSycl>
    {
        static auto freeVars(BlockSharedMemStGenericSycl const&) -> void
        {
            // shared memory block data will be reused
        }
    };
} // namespace alpaka::trait

#endif
