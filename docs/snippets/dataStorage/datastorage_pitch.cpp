/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

#define REQUIRE_MESSAGE(cond, msg)                                                                                    \
    do                                                                                                                \
    {                                                                                                                 \
        INFO(msg);                                                                                                    \
        REQUIRE(cond);                                                                                                \
    } while((void) 0, 0)

// BEGIN-DATASTORAGE-pitch-manual-calculation
template<typename T>
T get_element(std::byte* dataPtr, alpaka::concepts::Vector auto idx, alpaka::concepts::Vector auto pitches)
{
    // (idx * pitches).sum() is a dot product
    return *(reinterpret_cast<T*>(dataPtr + (idx * pitches).sum()));
}

// END-DATASTORAGE-pitch-manual-calculation

constexpr size_t columns = 5;
constexpr size_t rows = 3;
constexpr size_t matrices = 3;
constexpr size_t pitchRow = 2;
constexpr size_t rowWidthBytes = columns * sizeof(int32_t) + pitchRow;
constexpr size_t matrixWidthBytes = rows * rowWidthBytes + 22;

constexpr size_t memSizeByte2D = rowWidthBytes * rows;
alpaka::Vec pitches2D = alpaka::Vec{rowWidthBytes, sizeof(int32_t)};
std::array<std::byte, memSizeByte2D> mem2D;

constexpr size_t memSizeByte3D = matrixWidthBytes * matrices;
alpaka::Vec pitches3D = alpaka::Vec{matrixWidthBytes, rowWidthBytes, sizeof(int32_t)};
std::array<std::byte, memSizeByte3D> mem3D;

// this is a mock function which is required for the source code, which is displayed in the Sphinx doc
template<typename T>
auto allocHostBufferWithPadding(alpaka::concepts::Vector auto extents)
{
    if constexpr(extents.dim() == 2)
    {
        return alpaka::MdSpan{reinterpret_cast<T*>(mem2D.data()), extents, pitches2D};
    }
    else if constexpr(extents.dim() == 3)
    {
        return alpaka::MdSpan{reinterpret_cast<T*>(mem3D.data()), extents, pitches3D};
    }
    else
    {
        throw std::runtime_error("can only handle a 2D and 3D pitch");
    }
}

TEST_CASE("manually calculate 2D Pitch", "[docs]")
{
    REQUIRE(columns == 5);
    REQUIRE(rows == 3);
    REQUIRE(pitchRow == 2);
    REQUIRE(rowWidthBytes == 22);
    REQUIRE(memSizeByte2D == 66);
    REQUIRE(memSizeByte2D == rows * rowWidthBytes);

    std::fill(mem2D.begin(), mem2D.end(), std::byte{0});
    for(std::byte v : mem2D)
    {
        REQUIRE(v == std::byte{0});
    }

    // BEGIN-DATASTORAGE-pitch2D-example
    alpaka::Vec extents = alpaka::Vec{3u, 5u};
    auto buffer = allocHostBufferWithPadding<int32_t>(extents);

    REQUIRE(extents.x() == 5u);
    REQUIRE(extents.y() == 3u);

    // element size in byte (sizeof(int32_t))
    REQUIRE(buffer.getPitches().x() == 4u);
    // 5 elements with 4 bytes + 2 bytes padding
    REQUIRE(buffer.getPitches().y() == 22u);
    // END-DATASTORAGE-pitch2D-example

    int32_t value = 1;
    for(size_t row = 0u; row < rows; ++row)
    {
        for(size_t col = 0u; col < columns; ++col)
        {
            buffer[alpaka::Vec{row, col}] = value++;
        }
    }

    SECTION("validate with stepping byte by byte through linearized memory")
    {
        // a visualization of the 2D memory layout can be found in the documentation
        // https://alpaka.readthedocs.io/en/latest/advanced/datastorage.html#memory-layout-of-multidimensional-data-storage
        std::byte* memPtr = mem2D.data();
        // the byte postion in the 2D memory
        size_t offset = 0;
        // should be equal with the value, which was set over the MdSpan
        int32_t expected_value = 1;
        for(size_t row = 0u; row < rows; ++row)
        {
            // step 4 bytes each time because of the user values
            for(size_t col = 0u; col < columns; ++col)
            {
                int32_t value = *(reinterpret_cast<int32_t*>(memPtr + offset));
                REQUIRE_MESSAGE(value == expected_value, "coord: [" << row << ", " << col << "]\noffset: " << offset);
                expected_value += 1;
                offset += sizeof(int32_t);
            }
            // step 2 times one byte because of the row padding
            for(size_t pitch = 0u; pitch < pitchRow; ++pitch)
            {
                REQUIRE(*(memPtr + offset) == std::byte{0});
                offset += 1;
            }
        }
        // verify that we stepped through the whole memory
        REQUIRE(offset == memSizeByte2D);
    }

    SECTION("validate the memory with the get_element() function")
    {
        std::byte* memPtr = mem2D.data();
        int32_t expected_value = 1;
        for(size_t row = 0; row < rows; ++row)
        {
            for(size_t col = 0; col < columns; ++col)
            {
                REQUIRE_MESSAGE(
                    get_element<int32_t>(memPtr, alpaka::Vec{row, col}, pitches2D) == expected_value,
                    "coord: [" << row << ", " << col << "]");
                expected_value += 1;
            }
        }
    }
}

TEST_CASE("manual calculate 3D Pitch", "[docs]")
{
    REQUIRE(columns == 5);
    REQUIRE(rows == 3);
    REQUIRE(matrices == 3);
    REQUIRE(pitchRow == 2);
    REQUIRE(rowWidthBytes == 22);
    REQUIRE(matrixWidthBytes == 88);
    REQUIRE(memSizeByte3D == 264);
    REQUIRE(memSizeByte3D == (rows * rowWidthBytes + rowWidthBytes) * matrices);

    std::fill(mem3D.begin(), mem3D.end(), std::byte{0});

    for(std::byte v : mem3D)
    {
        REQUIRE(v == std::byte{0});
    }

    // BEGIN-DATASTORAGE-pitch3D-example
    alpaka::Vec extents = alpaka::Vec{3u, 3u, 5u};
    auto buffer = allocHostBufferWithPadding<int32_t>(extents);

    REQUIRE(extents.x() == 5u);
    REQUIRE(extents.y() == 3u);
    REQUIRE(extents.z() == 3u);

    // element size in byte (sizeof(int32_t))
    REQUIRE(buffer.getPitches().x() == 4u);
    // 5 elements with 4 bytes + 2 bytes padding
    REQUIRE(buffer.getPitches().y() == 22u);
    // 3 rows with user data and padding, each 22 bytes long + 22 bytes padding for empty row
    REQUIRE(buffer.getPitches().z() == 88u);
    // END-DATASTORAGE-pitch3D-example

    int32_t value = 1;
    for(size_t matrix = 0u; matrix < matrices; ++matrix)
    {
        for(size_t row = 0u; row < rows; ++row)
        {
            for(size_t col = 0u; col < columns; ++col)
            {
                buffer[alpaka::Vec{matrix, row, col}] = value++;
            }
        }
    }

    SECTION("validate with stepping byte by byte through linearized memory")
    {
        // a visualization of the 3D memory layout can be found in the documentation
        // https://alpaka.readthedocs.io/en/latest/advanced/datastorage.html#memory-layout-of-multidimensional-data-storage
        std::byte* memPtr = mem3D.data();
        // the byte postion in the 3D memory
        size_t offset = 0;
        // should be equal with the value, which was set over the MdSpan
        int32_t expected_value = 1;
        for(size_t matrix = 0u; matrix < matrices; ++matrix)
        {
            for(size_t row = 0u; row < rows; ++row)
            {
                // step 4 bytes each time because of the user values
                for(size_t col = 0u; col < columns; ++col)
                {
                    int32_t value = *(reinterpret_cast<int32_t*>(memPtr + offset));
                    REQUIRE_MESSAGE(
                        value == expected_value,
                        "coord: [" << matrix << ", " << row << ", " << col << "]\noffset: " << offset);
                    expected_value += 1;
                    offset += sizeof(int32_t);
                }
                // step 2 times one byte because of the row padding
                for(size_t pitch = 0u; pitch < pitchRow; ++pitch)
                {
                    REQUIRE_MESSAGE(
                        *(memPtr + offset) == std::byte{0},
                        "coord: [" << matrix << ", " << row << ", <pitch>]\noffset: " << offset);
                    offset += 1;
                }
            }
            // step 22 bytes to finish the matrix
            // the 22 bytes are a complete row with padding
            for(size_t pitch = 0u; pitch < rowWidthBytes; ++pitch)
            {
                REQUIRE_MESSAGE(
                    *(memPtr + offset) == std::byte{0},
                    "coord: [" << matrix << "<pitch> , <pitch>]\noffset: " << offset);
                offset += 1;
            }
        }
        // verify that we stepped through the whole memory
        REQUIRE(offset == memSizeByte3D);
    }

    SECTION("validate the memory with the get_element() function")
    {
        std::byte* memPtr = mem3D.data();
        int32_t expected_value = 1;
        for(size_t matrix = 0u; matrix < matrices; ++matrix)
        {
            for(size_t row = 0u; row < rows; ++row)
            {
                for(size_t col = 0u; col < columns; ++col)
                {
                    REQUIRE_MESSAGE(
                        get_element<int32_t>(memPtr, alpaka::Vec{matrix, row, col}, pitches3D) == expected_value,
                        "coord: [" << matrix << ", " << row << ", " << col << "]");
                    expected_value += 1;
                }
            }
        }
    }
}
