/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/idx/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>

#include <CL/sycl.hpp>

namespace alpaka
{
    namespace gb
    {
        //#############################################################################
        //! The SYCL accelerator ND index provider.
        template<
            typename TDim,
            typename TIdx>
        class IdxGbGenericSycl : public concepts::Implements<ConceptIdxGb, IdxGbGenericSycl<TDim, TIdx>>
        {
        public:
            using IdxGbBase = IdxGbGenericSycl;
            //-----------------------------------------------------------------------------
            IdxGbGenericSycl() = default;
            //-----------------------------------------------------------------------------
            explicit IdxGbGenericSycl(cl::sycl::nd_item<TDim::value> work_item)
            : my_item{work_item}
            {}
            //-----------------------------------------------------------------------------
            IdxGbGenericSycl(IdxGbGenericSycl const &) = default;
            //-----------------------------------------------------------------------------
            IdxGbGenericSycl(IdxGbGenericSycl &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbGenericSycl const & ) -> IdxGbGenericSycl & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxGbGenericSycl &&) -> IdxGbGenericSycl & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~IdxGbGenericSycl() = default;

            cl::sycl::nd_item<TDim::value> my_item;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<gb::IdxGbGenericSycl<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL accelerator grid block index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<gb::IdxGbGenericSycl<TDim, TIdx>, origin::Grid, unit::Blocks>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current block in the grid.
            template<typename TWorkDiv>
            static auto getIdx(gb::IdxGbGenericSycl<TDim, TIdx> const & idx, TWorkDiv const &) -> Vec<TDim, TIdx>
            {
                if constexpr(TDim::value == 1)
                    return Vec<TDim, TIdx>(static_cast<TIdx>(idx.my_item.get_group(0)));
                else if constexpr(TDim::value == 2)
                {
                    return Vec<TDim, TIdx>(static_cast<TIdx>(idx.my_item.get_group(1)),
                                           static_cast<TIdx>(idx.my_item.get_group(0)));
                }
                else
                {
                    return Vec<TDim, TIdx>(static_cast<TIdx>(idx.my_item.get_group(2)),
                                           static_cast<TIdx>(idx.my_item.get_group(1)),
                                           static_cast<TIdx>(idx.my_item.get_group(0)));
                }
            }
        };

        //#############################################################################
        //! The SYCL accelerator grid block index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<gb::IdxGbGenericSycl<TDim, TIdx>>
        {
            using type = TIdx;
        };
    }
}

#endif
