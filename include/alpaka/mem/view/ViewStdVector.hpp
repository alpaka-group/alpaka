/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

#include <vector>

namespace alpaka
{
    namespace traits
    {
        //#############################################################################
        //! The std::vector device type trait specialization.
        template<typename TElem, typename TAllocator>
        struct DevType<std::vector<TElem, TAllocator>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The std::vector device get trait specialization.
        template<typename TElem, typename TAllocator>
        struct GetDev<std::vector<TElem, TAllocator>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(std::vector<TElem, TAllocator> const& view) -> DevCpu
            {
                alpaka::ignore_unused(view);
                return getDevByIdx<PltfCpu>(0u);
            }
        };

        //#############################################################################
        //! The std::vector dimension getter trait specialization.
        template<typename TElem, typename TAllocator>
        struct DimType<std::vector<TElem, TAllocator>>
        {
            using type = DimInt<1u>;
        };

        //#############################################################################
        //! The std::vector memory element type get trait specialization.
        template<typename TElem, typename TAllocator>
        struct ElemType<std::vector<TElem, TAllocator>>
        {
            using type = TElem;
        };
    } // namespace traits
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector width get trait specialization.
            template<typename TElem, typename TAllocator>
            struct GetExtent<DimInt<0u>, std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(std::vector<TElem, TAllocator> const& extent)
                    -> Idx<std::vector<TElem, TAllocator>>
                {
                    return extent.size();
                }
            };
        } // namespace traits
    } // namespace extent
    namespace traits
    {
        //#############################################################################
        //! The std::vector native pointer get trait specialization.
        template<typename TElem, typename TAllocator>
        struct GetPtrNative<std::vector<TElem, TAllocator>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(std::vector<TElem, TAllocator> const& view) -> TElem const*
            {
                return view.data();
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(std::vector<TElem, TAllocator>& view) -> TElem*
            {
                return view.data();
            }
        };

        //#############################################################################
        //! The std::vector pitch get trait specialization.
        template<typename TElem, typename TAllocator>
        struct GetPitchBytes<DimInt<0u>, std::vector<TElem, TAllocator>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPitchBytes(std::vector<TElem, TAllocator> const& pitch)
                -> Idx<std::vector<TElem, TAllocator>>
            {
                return sizeof(TElem) * pitch.size();
            }
        };

        //#############################################################################
        //! The std::vector offset get trait specialization.
        template<typename TIdx, typename TElem, typename TAllocator>
        struct GetOffset<TIdx, std::vector<TElem, TAllocator>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getOffset(std::vector<TElem, TAllocator> const&)
                -> Idx<std::vector<TElem, TAllocator>>
            {
                return 0u;
            }
        };

        //#############################################################################
        //! The std::vector idx type trait specialization.
        template<typename TElem, typename TAllocator>
        struct IdxType<std::vector<TElem, TAllocator>>
        {
            using type = std::size_t;
        };
    } // namespace traits
} // namespace alpaka
