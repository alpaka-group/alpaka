/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <source_location>
#include <string_view>

namespace alpaka
{
    class UniqueId
    {
    public:
        static constexpr size_t getId(std::source_location const location = std::source_location::current())
        {
            return generate(location);
        }

    private:
        static constexpr size_t generate(std::source_location const& location)
        {
            size_t hash = 0xc6a4'a793'5bd1'e995;
            hashCombine(hash, location.file_name());
            hashCombine(hash, location.function_name());
            hashCombine(hash, location.line());
            hashCombine(hash, static_cast<size_t>(location.column()) << 32u);
            return hash;
        }

        static constexpr void hashCombine(size_t& seed, std::string_view value)
        {
            for(char c : value)
            {
                seed ^= static_cast<size_t>(c) + 0x9e37'79b9 + (seed << 6u) + (seed >> 2u);
            }
        }

        static constexpr void hashCombine(size_t& seed, size_t value)
        {
            seed ^= value + 0x9e37'79b9 + (seed << 6u) + (seed >> 2u);
        }
    };

    /** creates a unique id on any call
     *
     * If a class is storing the compile time id and the file of the class is included within two compile units the
     * id will be equal in both compile units.
     * The id is derived from the file name, function name, line, and column from where this method is called.
     * If this call is used to default set a template parameter of a class it will only generate once a unique number
     * not each time the class will be used.
     *
     * @param location The location is the base for the unique id. For the same location the same id is generated.
     * @return unique id
     */
    inline consteval size_t uniqueId(std::source_location const location = std::source_location::current())
    {
        return UniqueId::getId(location);
    }
} // namespace alpaka
