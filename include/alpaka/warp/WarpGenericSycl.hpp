/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED)

#include <alpaka/core/Unused.hpp>
#include <alpaka/warp/Traits.hpp>

#include <CL/sycl.hpp>

#include <cstdint>

namespace alpaka
{
    namespace warp
    {
        //#############################################################################
        //! The SYCL warp.
        template <typename TDim>
        class WarpGenericSycl : public concepts::Implements<ConceptWarp, WarpGenericSycl<TDim>>
        {
            friend struct traits::GetSize<WarpGenericSycl<TDim>>;
            friend struct traits::Activemask<WarpGenericSycl<TDim>>;
            friend struct traits::All<WarpGenericSycl<TDim>>;
            friend struct traits::Any<WarpGenericSycl<TDim>>;
            friend struct traits::Ballot<WarpGenericSycl<TDim>>;
        public:
            //-----------------------------------------------------------------------------
            WarpGenericSycl(cl::sycl::nd_item<TDim::value> my_item)
            : m_item{my_item}
            {}
            //-----------------------------------------------------------------------------
            WarpGenericSycl(WarpGenericSycl const &) = delete;
            //-----------------------------------------------------------------------------
            WarpGenericSycl(WarpGenericSycl &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WarpGenericSycl const &) -> WarpGenericSycl & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WarpGenericSycl &&) -> WarpGenericSycl & = delete;
            //-----------------------------------------------------------------------------
            ~WarpGenericSycl() = default;

        private:
            cl::sycl::nd_item<TDim::value> m_item;
        };

        namespace traits
        {
            //#############################################################################
            template<typename TDim>
            struct GetSize<
                WarpGenericSycl<TDim>>
            {
                //-----------------------------------------------------------------------------
                static auto getSize(warp::WarpGenericSycl<TDim> const & warp) -> std::int32_t
                {
                    const auto sub_group = warp.m_item.get_sub_group();
                    return static_cast<std::int32_t>(sub_group.get_local_range()[0]);
                }
            };

            //#############################################################################
            template<typename TDim>
            struct Activemask<
                WarpGenericSycl<TDim>>
            {
                //-----------------------------------------------------------------------------
                static auto activemask(warp::WarpGenericSycl<TDim> const & warp) -> std::uint32_t
                {
                    // No SYCL group algorithm for it, emulate via reduce
                    using namespace cl::sycl;

                    const auto sub_group = warp.m_item.get_sub_group();
                    const auto id = sub_group.get_local_linear_id();

                    // First step: Set the bit corresponding to our id to 1
                    const auto id_bitset = 1u << id;

                    // Second step: Reduction operation to combine all id_bitsets in the sub_group
                    return ONEAPI::reduce(sub_group, id_bitset, ONEAPI::bit_or<>{});
                }
            };

            //#############################################################################
            template<typename TDim>
            struct All<WarpGenericSycl<TDim>>
            {
                //-----------------------------------------------------------------------------
                static auto all(warp::WarpGenericSycl<TDim> const & warp, std::int32_t predicate) -> std::int32_t
                {
                    using namespace cl::sycl;

                    const auto sub_group = warp.m_item.get_sub_group();
                    return static_cast<std::int32_t>(ONEAPI::all_of(sub_group, static_cast<bool>(predicate))); 
                }
            };

            //#############################################################################
            template<typename TDim>
            struct Any<WarpGenericSycl<TDim>>
            {
                //-----------------------------------------------------------------------------
                static auto any(warp::WarpGenericSycl<TDim> const & warp, std::int32_t predicate) -> std::int32_t
                {
                    using namespace cl::sycl;

                    const auto sub_group = warp.m_item.get_sub_group();
                    return static_cast<std::int32_t>(ONEAPI::any_of(sub_group, static_cast<bool>(predicate)));
                }
            };

            //#############################################################################
            template<typename TDim>
            struct Ballot<
                WarpGenericSycl<TDim>>
            {
                //-----------------------------------------------------------------------------
                static auto ballot(warp::WarpGenericSycl<TDim> const & warp, std::int32_t predicate) -> std::uint32_t
                {
                    using namespace cl::sycl;

                    const auto sub_group = warp.m_item.get_sub_group();
                    const auto id = sub_group.get_local_linear_id();

                    // First step: Set the bit corresponding to our id to 1
                    const auto id_bitset = 1u << id;

                    // Second step: Check predicate and set bit accordingly
                    const auto pred_bit = (predicate != 0) ? 1u : 0u;
                    const auto pred_bitset = pred_bit << id;

                    // Third step: Create mask for our work item
                    const auto item_bitset = id_bitset & pred_bitset;

                    // Fourth step: Reduction operation to combine all item_bitsets in the sub_group
                    return ONEAPI::reduce(sub_group, item_bitset, ONEAPI::bit_or<>{});
                }
            };
        }
    }
}

#endif
