/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/algo/internal/scan.hpp"

// TODO: add assertion function for whether a device/api is compatible with a number of buffers
// (`onHost::isDataAccessible`)

namespace alpaka::onHost
{
    /** @brief For a scan of some size, this function returns the necessary buffer size in bytes.
     *
     * When multiple scans of the same extents are needed (for example in a loop), this function can be used to only
     * allocate an intermediate buffer once, removing alloc/free overhead. For unique scan calls, the buffer can be
     * omitted in the scan call, in which case it will be allocated and freed on the fly.
     *
     * @tparam T_Data The type of the data to be scanned.
     * @param extents The extents of the scan.
     * @return The size of the buffer to allocate in **number of bytes**.
     */
    template<typename T_Data>
    auto getScanBufferSize(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        return internal::scanBufferSize<T_Data>(extents);
    }

    /** @brief Perform an inclusive scan on the input data and write the result to the output data.
     *
     * @param queue The queue to enqueue to.
     * @param exec The executor to run with.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly. If you call this method
     * repeatedly, it is recommended to reuse the buffer whenever possible, or to provide a buffer allocated with
     * onHost::allocDeferred() to reduce the overhead of allocating and deallocating the buffer on each call.
     * @param outputVec The output data. To perform an in-place scan, use the overload with only one data object.
     * @param inputVec The input data. Can be const.
     *
     * @{
     */
    void inclusiveScan(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& buffer,
        alpaka::concepts::IMdSpan auto&& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::inclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(buffer),
                ALPAKA_FORWARD(outputVec),
                inputVec);
        }
        else
            internal::scan<internal::inclusiveScan>(queue, devAcc, exec, buffer, outputVec, inputVec);
    }

    void inclusiveScan(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::inclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(outputVec),
                inputVec);
        }
        else
            internal::scan<internal::inclusiveScan>(queue, devAcc, exec, ALPAKA_FORWARD(outputVec), inputVec);
    }

    /** @} */

    /** @brief Perform an inclusive scan on data in-place.
     *
     * @param queue The queue to enqueue to.
     * @param exec The executor to run with.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly. If you call this method
     * repeatedly, it is recommended to reuse the buffer whenever possible, or to provide a buffer allocated with
     * onHost::allocDeferred() to reduce the overhead of allocating and deallocating the buffer on each call.
     * @param dataVec The vector to scan, will be overwritten with the result.
     *
     * @{
     */
    void inclusiveScanInPlace(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& buffer,
        alpaka::concepts::IMdSpan auto&& dataVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::inclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(buffer),
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
        }
        else
            internal::scan<internal::inclusiveScan>(
                queue,
                devAcc,
                exec,
                buffer,
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
    }

    void inclusiveScanInPlace(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& dataVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::inclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
        }
        else
            internal::scan<internal::inclusiveScan>(
                queue,
                devAcc,
                exec,
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
    }

    /** @} */

    /** @brief Perform an exclusive scan on the input data and write the result to the output data.
     *
     * @param queue The queue to enqueue to.
     * @param exec The executor to run with.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly. If you call this method
     * repeatedly, it is recommended to reuse the buffer whenever possible, or to provide a buffer allocated with
     * onHost::allocDeferred() to reduce the overhead of allocating and deallocating the buffer on each call.
     * @param outputVec The output data. To perform an in-place scan, use the overload with only one data object.
     * @param inputVec The input data. Can be const.
     *
     * @{
     */
    void exclusiveScan(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& buffer,
        alpaka::concepts::IMdSpan auto&& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::exclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(buffer),
                ALPAKA_FORWARD(outputVec),
                inputVec);
        }
        else
            internal::scan<internal::exclusiveScan>(
                queue,
                devAcc,
                exec,
                ALPAKA_FORWARD(buffer),
                ALPAKA_FORWARD(outputVec),
                inputVec);
    }

    void exclusiveScan(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::exclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(outputVec),
                inputVec);
        }
        else
            internal::scan<internal::exclusiveScan>(queue, devAcc, exec, ALPAKA_FORWARD(outputVec), inputVec);
    }

    /** @} */

    /** @brief Perform an exclusive scan on data in-place.
     *
     * @param queue The queue to enqueue to.
     * @param exec The executor to run with.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly. If you call this method
     * repeatedly, it is recommended to reuse the buffer whenever possible, or to provide a buffer allocated with
     * onHost::allocDeferred() to reduce the overhead of allocating and deallocating the buffer on each call.
     * @param dataVec The vector to scan, will be overwritten with the result.
     *
     * @{
     */
    void exclusiveScanInPlace(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& buffer,
        alpaka::concepts::IMdSpan auto&& dataVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::exclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(buffer),
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
        }
        else
            internal::scan<internal::exclusiveScan>(
                queue,
                devAcc,
                exec,
                ALPAKA_FORWARD(buffer),
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
    }

    void exclusiveScanInPlace(
        auto const& queue,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::IMdSpan auto&& dataVec)
    {
        auto devAcc = queue.getDevice();
        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            internal::scan<internal::exclusiveScan>(
                queue,
                devAcc,
                defaultExecutor(devAcc),
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
        }
        else
            internal::scan<internal::exclusiveScan>(
                queue,
                devAcc,
                exec,
                ALPAKA_FORWARD(dataVec),
                ALPAKA_FORWARD(dataVec));
    }

    /** @} */
} // namespace alpaka::onHost
