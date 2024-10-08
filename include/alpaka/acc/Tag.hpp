/* Copyright 2023 Simeon Ehrig, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <iostream>
#include <type_traits>

namespace alpaka
{
    struct InterfaceTag
    {
    };
} // namespace alpaka

#define CREATE_ACC_TAG(tag_name)                                                                                      \
    struct tag_name : public alpaka::InterfaceTag                                                                     \
    {                                                                                                                 \
        static std::string get_name()                                                                                 \
        {                                                                                                             \
            return #tag_name;                                                                                         \
        }                                                                                                             \
    }

namespace alpaka
{
    CREATE_ACC_TAG(TagCpuOmp2Blocks);
    CREATE_ACC_TAG(TagCpuOmp2Threads);
    CREATE_ACC_TAG(TagCpuSerial);
    CREATE_ACC_TAG(TagCpuSycl);
    CREATE_ACC_TAG(TagCpuTbbBlocks);
    CREATE_ACC_TAG(TagCpuThreads);
    CREATE_ACC_TAG(TagFpgaSyclIntel);
    CREATE_ACC_TAG(TagGenericSycl);
    CREATE_ACC_TAG(TagGpuCudaRt);
    CREATE_ACC_TAG(TagGpuHipRt);
    CREATE_ACC_TAG(TagGpuSyclIntel);

    namespace concepts
    {
        template<typename T>
        concept Tag = requires {
            {
                T::get_name()
            } -> std::same_as<std::string>;
            requires std::default_initializable<T>;
            requires std::derived_from<T, alpaka::InterfaceTag>;
        };
    } // namespace concepts

    namespace trait
    {
        template<typename TAcc>
        struct AccToTag;

        template<concepts::Tag TTag, typename TDim, typename TIdx>
        struct TagToAcc;
    } // namespace trait

    //! \brief maps an acc type to a tag type
    //! \tparam TAcc alpaka acc type
    template<typename TAcc>
    using AccToTag = typename trait::AccToTag<TAcc>::type;

    //! \brief maps a tag type to an acc type
    //! \tparam TTag alpaka tag type
    //! \tparam TDim dimension of the mapped acc type
    //! \tparam TIdx index type of the mapped acc type
    template<concepts::Tag TTag, typename TDim, typename TIdx>
    using TagToAcc = typename trait::TagToAcc<TTag, TDim, TIdx>::type;

    template<typename TAcc, concepts::Tag... TTag>
    inline constexpr bool accMatchesTags = (std::is_same_v<alpaka::AccToTag<TAcc>, TTag> || ...);

    //! list of all available tags
    using AccTags = std::tuple<
        alpaka::TagCpuSerial,
        alpaka::TagCpuThreads,
        alpaka::TagCpuTbbBlocks,
        alpaka::TagCpuOmp2Blocks,
        alpaka::TagCpuOmp2Threads,
        alpaka::TagGpuCudaRt,
        alpaka::TagGpuHipRt,
        alpaka::TagCpuSycl,
        alpaka::TagFpgaSyclIntel,
        alpaka::TagGpuSyclIntel>;

} // namespace alpaka
