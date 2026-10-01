/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <thread>

namespace alpaka::test::event
{
    using namespace alpaka;

    namespace detail
    {
        // emulates sleeping even if there is no sleeep methods available
        inline ALPAKA_FN_ACC void alpakaSleep(
            [[maybe_unused]] int32_t* i,
            [[maybe_unused]] int32_t rounds,
            [[maybe_unused]] int32_t roundsGuard)
        {
#if ALPAKA_ARCH_PTX && __CUDA_ARCH__ >= 700
            __nanosleep(100000);
#elif !ALPAKA_LANG_SYCL && !ALPAKA_LANG_HIP && !ALPAKA_LANG_CUDA
            std::this_thread::sleep_for(std::chrono::microseconds(100u));
#else
            (*i) = 0;
            // Use a busy wait loop to avoid to fast polling
            for(; *i < roundsGuard; ++(*i))
            {
                if((*i) >= rounds)
                    break;
            }
#endif
        }

        /** Kernel waits for a signal to be finished.
         *
         * The kernel is stopping after a short time and report this not triggered exit.
         * This avoids that we deadlock.
         */
        struct KernelTestBussyWait
        {
            /**
             *
             * @param acc
             * @param triggerData
             * @param i
             * @param rounds
             * @param roundsGuard
             * @param status 1 if the kernel exists due to the host trigger else 0. Zero means the kernel stopped
             * because does not see the trigger data has changed.
             */
            ALPAKA_FN_ACC void operator()(
                onAcc::concepts::Acc auto const&,
                int volatile* triggerData,
                int* i,
                int32_t rounds,
                int32_t roundsGuard,
                int* status) const
            {
                constexpr uint32_t maxNumTriggerReads = 50000;
                uint32_t foo = 0;
                /* In principle, we could use an atomic load here, but tests showed that if we trigger with a second
                 * kernel with atomic operations, the second kernel is not running in parallel on the same backends.
                 * Using a volatile pointer showed to work best with all backends.
                 */
                while(*triggerData == 0)
                {
                    if(++foo >= maxNumTriggerReads)
                    {
                        *status = 0;
                        return;
                    }
                    // Use a busy wait loop to avoid to fast polling
                    alpakaSleep(i, rounds, roundsGuard);
                }
                *status = 1;
            }
        };

        struct UpdateValue
        {
            ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const&, int volatile* triggerData, int32_t value)
                const
            {
                *triggerData = value;
            }
        };

        inline bool detectConcurrentQueueHelper(auto& device)
        {
            auto triggerQueue = device.makeQueue();
            auto mainQueue = device.makeQueue();

            auto devCounter = onHost::alloc<int32_t>(device, 1);
            auto devTrigger = onHost::alloc<int32_t>(device, 1);
            auto status = onHost::allocMapped<int32_t>(device, 1);


            constexpr int waitLoopCount = 1000;
            onHost::fill(mainQueue, devCounter, 0);
            onHost::fill(mainQueue, devTrigger, 0);
            onHost::fill(mainQueue, status, -1);
            // wait that all data for the test is setup
            onHost::wait(mainQueue);
            mainQueue.enqueue(
                onHost::FrameSpec{Vec{1}, Vec{1}},
                KernelBundle{
                    KernelTestBussyWait{},
                    devTrigger.data(),
                    devCounter.data(),
                    waitLoopCount,
                    waitLoopCount + 10,
                    status.data()});

            // Uses a second kernel in a different queue to trigger the first kernel via a signal on device memory.
            triggerQueue.enqueue(
                onHost::FrameSpec{Vec{1}, Vec{1}},
                KernelBundle{UpdateValue{}, devTrigger.data(), 42});

            onHost::wait(mainQueue);
            onHost::wait(triggerQueue);
            return status[0] == 1;
        }

        inline bool mappedMemTriggerDetectionHelper(auto& device)
        {
            auto mainQueue = device.makeQueue();

            auto devCounter = onHost::alloc<int32_t>(device, 1);
            auto trigger = onHost::allocMapped<int32_t>(device, 1);
            auto status = onHost::allocMapped<int32_t>(device, 1);


            constexpr int waitLoopCount = 1000;
            onHost::fill(mainQueue, devCounter, 0);
            trigger[0] = 0;
            onHost::fill(mainQueue, status, -1);
            // wait that all data for the test is set up
            onHost::wait(mainQueue);
            mainQueue.enqueue(
                onHost::FrameSpec{Vec{1}, Vec{1}},
                KernelBundle{
                    KernelTestBussyWait{},
                    trigger.data(),
                    devCounter.data(),
                    waitLoopCount,
                    waitLoopCount + 10,
                    status.data()});

            // trigger via mapped memory
            trigger[0] = 42;

            onHost::wait(mainQueue);
            return status[0] == 1;
        }

        struct KernelBlockUntilReady
        {
            ALPAKA_FN_ACC void operator()(
                onAcc::concepts::Acc auto const&,
                int volatile* triggerData,
                int* i,
                int rounds,
                int roundsGuard) const
            {
                while(*triggerData == 0)
                {
                    alpakaSleep(i, rounds, roundsGuard);
                }
            }
        };
    } // namespace detail

    /** Checks if device queue can execute concurrent work.
     *
     * Some native api's alpaka is using have only a single hardware/ord driver limited compute queue, in this case all
     * tasks enqueued in user created independent queues will be in FIFO mode.
     * The test checks only if two queues can run concurrent, it not necessary says more than two queues can run
     * concurrent.
     *
     * @param device device which is tested
     * @param maxRepetitions number of rounds the test is executed to avoid false positives
     * @return true if the device support concurrent queues, else false. @attention: false is returned even if the
     * device is supporting concurrent queues, the reason is that the detection works with hard coded number of
     * retries. This is required to avoid triggering the watchdog terminating long-running kernels on some systems. If
     * one of the test repetitions reports that concurrent queues are not supported the result will be false.
     */
    inline bool detectConcurrentQueue(auto& device, int maxRepetitions = 3)
    {
        bool result = true;
        for(int i = 0; i < maxRepetitions && result; ++i)
            result = result && detail::detectConcurrentQueueHelper(device);
        return result;
    }

    /** Checks if mapped memory can be used to trigger a kernel.
     *
     * The test checks if a kernel can be triggered by writing into mapped memory.
     * The test is executed multiple times to avoid false positives.
     *  @return true if the device support concurrent queues, else false
     */
    inline bool mappedMemTriggerDetection(auto& device, int maxRepetitions = 3)
    {
        bool result = true;
        for(int i = 0; i < maxRepetitions && result; ++i)
            result = result && detail::mappedMemTriggerDetectionHelper(device);
        return result;
    }

    // Checks if the device supports concurrent queue execution and if we can send via mapped memory information to a
    // running kernel.
    inline bool checkIfDeviceCanExecuteEventTests(auto& device, int maxRepetitions = 3)
    {
        return mappedMemTriggerDetection(device, maxRepetitions) && detectConcurrentQueue(device, maxRepetitions);
    }

    /** Emulates a kernel which runs until an event is triggered on host side to release the kernel.
     *
     * Use mapped memory to send signals to the blocking kernels to release them.
     */
    template<typename T_Device>
    struct TriggerKernel
    {
        TriggerKernel() = default;

        TriggerKernel(T_Device& device)
            : m_CounterDev{onHost::alloc<int32_t>(device, 1)}
            , m_triggerData{onHost::allocMapped<int32_t>(device, 1)}
            , m_preEvent{device.makeEvent()}
            , m_postEvent{device.makeEvent()}
        {
            auto initQueue = device.makeQueue();
            onHost::fill(initQueue, m_CounterDev, 0);
            onHost::fill(initQueue, m_triggerData, 0);
            onHost::wait(initQueue);
        }

        // set event to ready state
        void trigger()
        {
            for(int i = 0; i < 10; ++i)
            {
                m_triggerData[0] = 42;
                // Give alpaka time to update into the new state, process all events and tasks.
                std::this_thread::sleep_for(std::chrono::milliseconds(500u));
                if(isRunning())
                    break;
                else
                {
                    WARN("trigger had no effect, re-trigger " << i + 1 << " of " << 10);
                }
            }
        }

        void submit(auto& queue)
        {
            constexpr int waitLoopCount = 10000;

            queue.enqueue(m_preEvent);
            queue.enqueue(
                onHost::FrameSpec{Vec{1}, Vec{1}},
                KernelBundle{
                    detail::KernelBlockUntilReady{},
                    m_triggerData.data(),
                    m_CounterDev.data(),
                    waitLoopCount,
                    waitLoopCount + 10});
            queue.enqueue(m_postEvent);
        }

        /** Checks if the kernel execution is NOT finished
         *
         * @attention Do not negate the return value for the positive test, use assumeComplete() instead.
         *
         * @return true if the kernel is NOT finished yet, else false.
         */
        bool assumeNotComplete() const
        {
            return !m_postEvent.isComplete();
        }

        /** Assume the kernel is executed
         *
         * Due to the implementation if the host side triggerable kernel for this test the simple status check
         * can provide a false negative. This method is polling few times to guarantee the information arrives the
         * kernel even if the tested device is executing other tasks too.
         *
         * @attention Do not negate the return value for the negative test, use assumeNotComplete() instead.
         *
         * @param repeat Number of repetitions of the status check.
         * @param ms Milliseconds to wait between the status checks.
         * @return Return true if kernel is executed, at least after `repeat` status test calls, else false.
         */
        bool assumeComplete(uint32_t repeat = 13, size_t ms = 200) const
        {
            bool result = false;
            for(uint32_t i = 0; i < repeat; ++i)
            {
                result = m_postEvent.isComplete();
                if(result == false)
                    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                else
                    break;
            }
            return result;
        }

        // inform if the emulated kernel is scheduled by the device and therefore can be triggered and tested for
        // completeness
        bool isRunning() const
        {
            return m_preEvent.isComplete();
        }

        void wait() const
        {
            return onHost::wait(m_postEvent);
        }

        ALPAKA_TYPEOF(onHost::alloc<int32_t>(std::declval<T_Device>(), 1)) m_CounterDev;
        ALPAKA_TYPEOF(onHost::allocMapped<int32_t>(std::declval<T_Device>(), 1)) m_triggerData;
        onHost::Event<T_Device> m_preEvent;
        onHost::Event<T_Device> m_postEvent;
    };
} // namespace alpaka::test::event
