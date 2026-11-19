/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/mem/concepts/ExpectedValueType.hpp"
#include "alpaka/mem/concepts/IDataSource.hpp"
#include "alpaka/trait.hpp"

#include <concepts>

namespace alpaka::concepts
{
    namespace impl
    {
        /** @brief Interface concept for objects describing multidimensional memory access.
         *
         * @details
         *
         * An object of type `alpaka::mdspan` does not store any information about the storage location, e.g., whether
         * the memory is located on a CPU or a GPU. The interface corresponds to that of a standard library container
         * with continuous memory, but has some differences to support multidimensional memory. For example, instead of
         * the member function `size()`, which returns the 1D size, `alpaka::mdspan` like objects provides the function
         *`getExtents()`, which returns the size of each dimension.
         *
         * @param t Object of type `alpaka::mdspan`. May or may not have a const modifier.
         * @param mut_t Mutable object of type `alpaka::mdspan`. Does not have a const modifier.
         * @param const_t Constant object of type `alpaka::mdspan`. Does have a const modifier.
         * @param vec Vector with the same number of elements as the dimension of the `alpaka::mdspan` like object.
         * Used to call the access operator.
         *
         *  @section components Components
         *
         * An `alpaka::mdspan` like object contains 4 components:
         * - A pointer to the actual memory.
         * - An extents object that describes the number of dimensions and their respective sizes.
         * - A pitch object that specifies how many bytes are required to jump to the next element in each dimension.
         * - An alignment object that describes how the elements are aligned in memory, see:
         * alpaka::concepts::Alignment
         *
         * @section Member types
         * - <b>T::reference</b>: The element reference type is either const or non-const, depending on
         *`T::value_type`.
         * - <b>T::const_reference</b>: The constant reference type for an element. Always const.
         * - <b>T::pointer</b>: The element pointer type is either const or non-const, depending on
         *`T::value_type`.
         * - <b>T::const_pointer</b>: The constant pointer type for an element. Always pointer-to-const.
         **/
        template<typename T, typename T_Mut, typename T_Const>
        concept IMdSpan
            = requires(T t, T_Mut mutT, T_Const constT, alpaka::Vec<typename T::index_type, T::dim()> vec) {
                  requires IDataSource<T>;

                  typename T::reference;
                  typename T::const_reference;
                  typename T::pointer;
                  typename T::const_pointer;

                  { *mutT } -> std::same_as<typename T::reference>;
                  { *constT } -> std::same_as<typename T::const_reference>;
                  { mutT.data() } -> std::same_as<typename T::pointer>;
                  { constT.data() } -> std::same_as<typename T::const_pointer>;

                  { mutT[vec] } -> std::same_as<typename T::reference>;
                  { constT[vec] } -> std::same_as<typename T::const_reference>;
                  // only if MdSpan like object is 1D, the access operator with an integral is available
                  requires(T::dim() != 1u) || (T::dim() == 1u && requires {
                              { mutT[typename T::index_type{0}] } -> std::same_as<typename T::reference>;
                          });
                  requires(T::dim() != 1u) || (T::dim() == 1u && requires {
                              { constT[typename T::index_type{0}] } -> std::same_as<typename T::const_reference>;
                          });
                  /// @todo add getSlice, getConstSlice and getView, getConstView functions
              };

    } // namespace impl

    /** @brief Interface concept for objects describing multidimensional memory access.
     *
     * @details
     * An object of type `alpaka::mdspan` does not store any information about the storage location, e.g., whether
     * the memory is located on a CPU or a GPU.
     *
     * @attention Use `alpaka::IMdSpan` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IMdSpan.
     **/
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IMdSpan = requires {
        requires impl::IMdSpan<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
