/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
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

#include <iostream>
#include <typeinfo>
#include <vector>

//! A kernel using atomicOp, syncBlockThreads, getDynSharedMem, getIdx, getWorkDiv and global memory to compute a
//! (useless) result. \tparam TnumUselessWork The number of useless calculations done in each kernel execution.
template<typename TTnumUselessWork, typename TVal>
class SharedMemKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TVal* const pui_block_ret_vals) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;

        static_assert(alpaka::Dim<TAcc>::value == 1, "The SharedMemKernel expects 1-dimensional indices!");

        // The number of threads in this block.
        Idx const block_thread_count(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // Get the dynamically allocated shared memory.
        TVal* const p_block_shared(alpaka::getDynSharedMem<TVal>(acc));

        // Calculate linearized index of the thread in the block.
        Idx const block_thread_idx1d(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);


        // Fill the shared block with the thread ids [1+X, 2+X, 3+X, ..., #Threads+X].
        auto sum1 = static_cast<TVal>(block_thread_idx1d + 1);
        for(TVal i(0); i < static_cast<TVal>(TTnumUselessWork::value); ++i)
        {
            sum1 += i;
        }
        p_block_shared[block_thread_idx1d] = sum1;


        // Synchronize all threads because now we are writing to the memory again but inverse.
        alpaka::syncBlockThreads(acc);

        // Do something useless.
        auto sum2 = static_cast<TVal>(block_thread_idx1d);
        for(TVal i(0); i < static_cast<TVal>(TTnumUselessWork::value); ++i)
        {
            sum2 -= i;
        }
        // Add the inverse so that every cell is filled with [#Threads, #Threads, ..., #Threads].
        p_block_shared[(block_thread_count - 1) - block_thread_idx1d] += sum2;


        // Synchronize all threads again.
        alpaka::syncBlockThreads(acc);

        // Now add up all the cells atomically and write the result to cell 0 of the shared memory.
        if(block_thread_idx1d > 0)
        {
            alpaka::atomicAdd(acc, &p_block_shared[0], p_block_shared[block_thread_idx1d]);
        }


        alpaka::syncBlockThreads(acc);

        // Only master writes result to global memory.
        if(block_thread_idx1d == 0)
        {
            // Calculate linearized block id.
            Idx const grid_block_idx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);

            pui_block_ret_vals[grid_block_idx] = p_block_shared[0];
        }
    }
};

namespace alpaka
{
    namespace traits
    {
        //! The trait for getting the size of the block shared dynamic memory for a kernel.
        template<typename TTnumUselessWork, typename TVal, typename TAcc>
        struct BlockSharedMemDynSizeBytes<SharedMemKernel<TTnumUselessWork, TVal>, TAcc>
        {
            //! \return The size of the shared memory allocated for a block.
            template<typename TVec, typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto get_block_shared_mem_dyn_size_bytes(
                SharedMemKernel<TTnumUselessWork, TVal> const& shared_mem_kernel,
                TVec const& block_thread_extent,
                TVec const& thread_elem_extent,
                TArgs&&...) -> std::size_t
            {
                alpaka::ignore_unused(shared_mem_kernel);
                return static_cast<std::size_t>(block_thread_extent.prod() * thread_elem_extent.prod()) * sizeof(TVal);
            }
        };
    } // namespace traits
} // namespace alpaka

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("sharedMem", "[sharedMem]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    Idx const num_elements = 1u << 16u;

    using Val = std::int32_t;
    using TnumUselessWork = std::integral_constant<Idx, 100>;

    using DevAcc = alpaka::Dev<Acc>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;


    // Create the kernel function object.
    SharedMemKernel<TnumUselessWork, Val> kernel;

    // Select a device to execute on.
    auto const dev_acc = alpaka::getDevByIdx<PltfAcc>(0u);

    // Get a queue on this device.
    QueueAcc queue(dev_acc);

    // Set the grid blocks extent.
    alpaka::WorkDivMembers<Dim, Idx> const work_div(alpaka::getValidWorkDiv<Acc>(
        dev_acc,
        num_elements,
        static_cast<Idx>(1u),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted));

    std::cout << "SharedMemKernel("
              << " accelerator: " << alpaka::getAccName<Acc>() << ", kernel: " << typeid(kernel).name()
              << ", workDiv: " << work_div << ")" << std::endl;

    Idx const grid_blocks_count(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(work_div)[0u]);
    Idx const block_thread_count(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(work_div)[0u]);

    // An array for the return values calculated by the blocks.
    std::vector<Val> block_ret_vals(static_cast<std::size_t>(grid_blocks_count));

    // Allocate accelerator buffers and copy.
    Idx const result_elem_count(grid_blocks_count);
    auto block_ret_vals_acc = alpaka::allocBuf<Val, Idx>(dev_acc, result_elem_count);
    alpaka::memcpy(queue, block_ret_vals_acc, block_ret_vals, result_elem_count);

    // Create the kernel execution task.
    auto const task_kernel = alpaka::createTaskKernel<Acc>(work_div, kernel, alpaka::getPtrNative(block_ret_vals_acc));

    // Profile the kernel execution.
    std::cout << "Execution time: " << alpaka::test::integ::measureTaskRunTimeMs(queue, task_kernel) << " ms"
              << std::endl;

    // Copy back the result.
    alpaka::memcpy(queue, block_ret_vals, block_ret_vals_acc, result_elem_count);

    // Wait for the queue to finish the memory operation.
    alpaka::wait(queue);

    // Assert that the results are correct.
    auto const correct_result(static_cast<Val>(block_thread_count * block_thread_count));

    bool result_correct(true);
    for(Idx i(0); i < grid_blocks_count; ++i)
    {
        auto const val(block_ret_vals[static_cast<std::size_t>(i)]);
        if(val != correct_result)
        {
            std::cerr << "blockRetVals[" << i << "] == " << val << " != " << correct_result << std::endl;
            result_correct = false;
        }
    }

    REQUIRE(result_correct);
}
