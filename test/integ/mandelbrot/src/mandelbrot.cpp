/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/MeasureKernelRunTime.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <typeinfo>

//#define ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING  // Define this to enable the continuous color mapping.

//! Complex Number.
template<typename TT>
class SimpleComplex
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC SimpleComplex(TT const& a, TT const& b) : m_r(a), m_i(b)
    {
    }
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_INLINE
    ALPAKA_FN_HOST_ACC auto abs_sq() const -> TT
    {
        return m_r * m_r + m_i * m_i;
    }
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator*(SimpleComplex const& a) -> SimpleComplex
    {
        return SimpleComplex(m_r * a.m_r - m_i * a.m_i, m_i * a.m_r + m_r * a.m_i);
    }
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator*(float const& a) -> SimpleComplex
    {
        return SimpleComplex(m_r * a, m_i * a);
    }
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator+(SimpleComplex const& a) -> SimpleComplex
    {
        return SimpleComplex(m_r + a.m_r, m_i + a.m_i);
    }
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_HOST_ACC auto operator+(float const& a) -> SimpleComplex
    {
        return SimpleComplex(m_r + a, m_i);
    }


    TT m_r;
    TT m_i;
};

//! A Mandelbrot kernel.
class MandelbrotKernel
{
public:
#ifndef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
    ALPAKA_FN_HOST_ACC MandelbrotKernel()
    {
        // Banding can be prevented by a continuous color functions.
        m_colors[0u] = convert_rgb_single_to_bgra(66, 30, 15);
        m_colors[1u] = convert_rgb_single_to_bgra(25, 7, 26);
        m_colors[2u] = convert_rgb_single_to_bgra(9, 1, 47);
        m_colors[3u] = convert_rgb_single_to_bgra(4, 4, 73);
        m_colors[4u] = convert_rgb_single_to_bgra(0, 7, 100);
        m_colors[5u] = convert_rgb_single_to_bgra(12, 44, 138);
        m_colors[6u] = convert_rgb_single_to_bgra(24, 82, 177);
        m_colors[7u] = convert_rgb_single_to_bgra(57, 125, 209);
        m_colors[8u] = convert_rgb_single_to_bgra(134, 181, 229);
        m_colors[9u] = convert_rgb_single_to_bgra(211, 236, 248);
        m_colors[10u] = convert_rgb_single_to_bgra(241, 233, 191);
        m_colors[11u] = convert_rgb_single_to_bgra(248, 201, 95);
        m_colors[12u] = convert_rgb_single_to_bgra(255, 170, 0);
        m_colors[13u] = convert_rgb_single_to_bgra(204, 128, 0);
        m_colors[14u] = convert_rgb_single_to_bgra(153, 87, 0);
        m_colors[15u] = convert_rgb_single_to_bgra(106, 52, 3);
    }
#endif

    //! \param acc The accelerator to be executed on.
    //! \param pColors The output image.
    //! \param numRows The number of rows in the image
    //! \param numCols The number of columns in the image.
    //! \param pitchBytes The pitch in bytes.
    //! \param fMinR The left border.
    //! \param fMaxR The right border.
    //! \param fMinI The bottom border.
    //! \param fMaxI The top border.
    //! \param maxIterations The maximum number of iterations.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        const std::uint32_t* const p_colors,
        std::uint32_t const& num_rows,
        std::uint32_t const& num_cols,
        std::uint32_t const& pitch_bytes,
        float const& f_min_r,
        float const& f_max_r,
        float const& f_min_i,
        float const& f_max_i,
        std::uint32_t const& max_iterations) const -> void
    {
        static_assert(alpaka::Dim<TAcc>::value == 2, "The MandelbrotKernel expects 2-dimensional indices!");

        auto const grid_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const& grid_thread_idx_x = grid_thread_idx[1u];
        auto const& grid_thread_idx_y = grid_thread_idx[0u];

        if((grid_thread_idx_y < num_rows) && (grid_thread_idx_x < num_cols))
        {
            SimpleComplex<float> c(
                (f_min_r + (static_cast<float>(grid_thread_idx_x) / float(num_cols - 1) * (f_max_r - f_min_r))),
                (f_min_i + (static_cast<float>(grid_thread_idx_y) / float(num_rows - 1) * (f_max_i - f_min_i))));

            auto const iteration_count = iterate_mandelbrot(c, max_iterations);

            auto const p_colors_row = p_colors + ((grid_thread_idx_y * pitch_bytes) / sizeof(std::uint32_t));
            p_colors_row[grid_thread_idx_x] =
#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
                iterationCountToContinousColor(iterationCount, maxIterations);
#else
                iteration_count_to_repeated_color(iteration_count);
#endif
        }
    }
    //! \return The number of iterations until the Mandelbrot iteration with the given Value reaches the absolute value
    //! of 2.
    //!     Only does maxIterations steps and returns maxIterations if the value would be higher.
    ALPAKA_FN_ACC static auto iterate_mandelbrot(SimpleComplex<float> const& c, std::uint32_t const& max_iterations)
        -> std::uint32_t
    {
        SimpleComplex<float> z(0.0f, 0.0f);
        for(std::uint32_t iterations(0); iterations < max_iterations; ++iterations)
        {
            z = z * z + c;
            if(z.abs_sq() > 4.0f)
            {
                return iterations;
            }
        }
        return max_iterations;
    }

    ALPAKA_FN_HOST_ACC static auto convert_rgb_single_to_bgra(
        std::uint32_t const& r,
        std::uint32_t const& g,
        std::uint32_t const& b) -> std::uint32_t
    {
        return 0xFF000000 | (r << 16) | (g << 8) | b;
    }

#ifdef ALPAKA_MANDELBROT_TEST_CONTINOUS_COLOR_MAPPING
    //! This uses a simple mapping from iteration count to colors.
    //! This leads to banding but allows a all pixels to be colored.
    ALPAKA_NO_HOST_ACC_WARNING
    ALPAKA_FN_ACC static auto iterationCountToContinousColor(
        std::uint32_t const& iterationCount,
        std::uint32_t const& maxIterations) -> std::uint32_t
    {
        // Map the iteration count on the 0..1 interval.
        float const t(static_cast<float>(iterationCount) / static_cast<float>(maxIterations));
        float const oneMinusT(1.0f - t);
        // Use some modified Bernstein polynomials for r, g, b.
        std::uint32_t const r(static_cast<std::uint32_t>(9.0f * oneMinusT * t * t * t * 255.0f));
        std::uint32_t const g(static_cast<std::uint32_t>(15.0f * oneMinusT * oneMinusT * t * t * 255.0f));
        std::uint32_t const b(static_cast<std::uint32_t>(8.5f * oneMinusT * oneMinusT * oneMinusT * t * 255.0f));
        return convertRgbSingleToBgra(r, g, b);
    }
#else
    //! This uses a simple mapping from iteration count to colors.
    //! This leads to banding but allows a all pixels to be colored.
    ALPAKA_FN_ACC auto iteration_count_to_repeated_color(std::uint32_t const& iteration_count) const -> std::uint32_t
    {
        return m_colors[iteration_count % 16];
    }

    std::uint32_t m_colors[16]{};
#endif
};

//! Writes the buffer color data to a file.
template<typename TBuf>
auto write_tga_color_image(std::string const& file_name, TBuf const& buf_rgba) -> void
{
    static_assert(alpaka::Dim<TBuf>::value == 2, "The buffer has to be 2 dimensional!");
    static_assert(std::is_integral<alpaka::Elem<TBuf>>::value, "The buffer element type has to be integral!");

    // The width of the input buffer is in input elements.
    auto const buf_width_elems = alpaka::extent::getWidth(buf_rgba);
    auto const buf_width_bytes = buf_width_elems * sizeof(alpaka::Elem<TBuf>);
    // The row width in bytes has to be dividable by 4 Bytes (RGBA).
    ALPAKA_ASSERT(buf_width_bytes % sizeof(std::uint32_t) == 0);
    // The number of colors in a row.
    auto const buf_width_colors = buf_width_bytes / sizeof(std::uint32_t);
    ALPAKA_ASSERT(buf_width_colors >= 1);
    auto const buf_height_colors = alpaka::extent::getHeight(buf_rgba);
    ALPAKA_ASSERT(buf_height_colors >= 1);
    auto const buf_pitch_bytes = alpaka::getPitchBytes<alpaka::Dim<TBuf>::value - 1u>(buf_rgba);
    ALPAKA_ASSERT(buf_pitch_bytes >= buf_width_bytes);

    std::ofstream ofs(file_name, std::ofstream::out | std::ofstream::binary);
    if(!ofs.is_open())
    {
        throw std::invalid_argument("Unable to open file: " + file_name);
    }

    // Write tga image header.
    ofs.put(0x00); // Number of Characters in Identification Field.
    ofs.put(0x00); // Color Map Type.
    ofs.put(0x02); // Image Type Code.
    ofs.put(0x00); // Color Map Origin.
    ofs.put(0x00);
    ofs.put(0x00); // Color Map Length.
    ofs.put(0x00);
    ofs.put(0x00); // Color Map Entry Size.
    ofs.put(0x00); // X Origin of Image.
    ofs.put(0x00);
    ofs.put(0x00); // Y Origin of Image.
    ofs.put(0x00);
    ofs.put(static_cast<char>(buf_width_colors & 0xFFu)); // Width of Image.
    ofs.put(static_cast<char>((buf_width_colors >> 8) & 0xFFu));
    ofs.put(static_cast<char>(buf_height_colors & 0xFFu)); // Height of Image.
    ofs.put(static_cast<char>((buf_height_colors >> 8) & 0xFFu));
    ofs.put(0x20); // Image Pixel Size.
    ofs.put(0x20); // Image Descriptor Byte.

    // Write the data.
    auto const* p_data(reinterpret_cast<char const*>(alpaka::getPtrNative(buf_rgba)));
    // If there is no padding, we can directly write the whole buffer data ...
    if(buf_pitch_bytes == buf_width_bytes)
    {
        ofs.write(p_data, static_cast<std::streamsize>(buf_width_bytes * buf_height_colors));
    }
    // ... else we have to write row by row.
    else
    {
        for(auto row = decltype(buf_height_colors)(0); row < buf_height_colors; ++row)
        {
            ofs.write(p_data + buf_pitch_bytes * row, static_cast<std::streamsize>(buf_width_bytes));
        }
    }
}

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<2u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("mandelbrot", "[mandelbrot]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

#ifdef ALPAKA_CI
    Idx const imageSize(1u << 5u);
#else
    Idx const image_size(1u << 10u);
#endif
    Idx const num_rows(image_size);
    Idx const num_cols(image_size);
    float const f_min_r(-2.0f);
    float const f_max_r(+1.0f);
    float const f_min_i(-1.2f);
    float const f_max_i(+1.2f);
    Idx const max_iterations(300u);

    using Val = std::uint32_t;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;
    using PltfHost = alpaka::PltfCpu;

    // Create the kernel function object.
    MandelbrotKernel kernel;

    // Get the host device.
    auto const dev_host = alpaka::getDevByIdx<PltfHost>(0u);

    // Select a device to execute on.
    auto const dev_acc = alpaka::getDevByIdx<PltfAcc>(0u);

    // Get a queue on this device.
    QueueAcc queue(dev_acc);

    alpaka::Vec<Dim, Idx> const extent(static_cast<Idx>(num_rows), static_cast<Idx>(num_cols));

    // Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::WorkDivMembers<Dim, Idx> const work_div(alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        extent,
        alpaka::Vec<Dim, Idx>::ones(),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout << "MandelbrotKernel("
              << " numRows:" << num_rows << ", numCols:" << num_cols << ", maxIterations:" << max_iterations
              << ", accelerator: " << alpaka::getAccName<Acc>() << ", kernel: " << typeid(kernel).name()
              << ", workDiv: " << work_div << ")" << std::endl;

    // allocate host memory
    auto buf_color_host = alpaka::allocBuf<Val, Idx>(dev_host, extent);

    // Allocate the buffer on the accelerator.
    auto buf_color_acc = alpaka::allocBuf<Val, Idx>(dev_acc, extent);

    // Copy Host -> Acc.
    alpaka::memcpy(queue, buf_color_acc, buf_color_host, extent);

    // Create the kernel execution task.
    auto const task_kernel = alpaka::createTaskKernel<Acc>(
        work_div,
        kernel,
        alpaka::getPtrNative(buf_color_acc),
        num_rows,
        num_cols,
        alpaka::getPitchBytes<1u>(buf_color_acc),
        f_min_r,
        f_max_r,
        f_min_i,
        f_max_i,
        max_iterations);

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queue, task_kernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queue, buf_color_host, buf_color_acc, extent);

    // Wait for the queue to finish the memory operation.
    alpaka::wait(queue);

    // Write the image to a file.
    std::string file_name(
        "mandelbrot" + std::to_string(num_cols) + "x" + std::to_string(num_rows) + "_" + alpaka::getAccName<Acc>()
        + ".tga");
    std::replace(file_name.begin(), file_name.end(), '<', '_');
    std::replace(file_name.begin(), file_name.end(), '>', '_');
    write_tga_color_image(file_name, buf_color_host);
}
