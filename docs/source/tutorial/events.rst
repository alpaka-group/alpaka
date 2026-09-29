Events and Synchronization
==========================

As soon as you need to know if a task in a queue is already executed or you think about task parallelism you need events.
This allows to describe dependencies between tasks in different queues without blocking the host thread.
In *alpaka*, queues describe execution order, and events describe dependencies between queues.

Queue Rules
-----------

- Operations inside one queue execute in FIFO order.
- Different queues may run independently.
- ``onHost::wait(queue)`` waits until all work in that queue is complete.
- ``onHost::wait(event)`` waits until the event has been processed which means that all previously enqueued tasks are completed.
- ``queue1.waitFor(event)`` inserts a dependency so work enqueued in ``queue1`` afterwards starts only after the event is reached.

Use Cases
---------

Multiple queues are primarily used in the following scenarios:

- Host and device queues: Run independent tasks on the host CPU and the device (e.g., a GPU) and synchronize both devices.
- Many queues for many devices: For example, in multi-GPU systems, each GPU requires its own queue.
- Many queues for a single device: Enables better utilization of a single device. The performance benefits depend on the :ref:`API`.

Creating and Using an Event
---------------------------

In the following example, we create two queues (``queue0`` and ``queue1``).
Both execute functions on the host via ``enqueueHostFn()``.
Without synchronization between the queues using events, a race condition is possible.
It is possible that ``queue1`` increments the value ``valueA`` before ``valueA`` is set to ``41`` in ``queue0``.

  .. literalinclude:: ../../snippets/example/160_events.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-event
    :end-before: END-TUTORIAL-event
    :dedent:

Measuring Queued Work
---------------------

Timing is an explicit queue capability and is disabled by default. Enable it by adding ``timing::enabled`` to the
``onHost::QueuePolicyList`` passed to ``makeQueue``. Events use ``onHost::EventPolicyList`` and can independently be
constructed with ``timing::enabled`` or ``timing::disabled``. Passing an individual policy directly is shorthand for
constructing the corresponding policy list. ``queue.makeEvent()`` inherits the queue's timing capability and accepts
additional non-timing event policies.

A timing-enabled event can only be enqueued on a timing-enabled queue. This rule is the same for every API, even where
a native backend would technically allow a timed event on an ordinary queue.

The elapsed time is obtained from backend event information. CUDA and HIP use native event timestamps, SYCL uses
profiling timestamps from its profiling-enabled queue, and the host backend records a monotonic timestamp when each
event marker is reached.

  .. literalinclude:: ../../snippets/example/165_eventTiming.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-eventTiming
    :end-before: END-TUTORIAL-eventTiming
    :dedent:

``onHost::getElapsedTime()`` returns a ``std::chrono::duration<double>``. The measured interval excludes the timing
markers themselves and includes all queue work between them. Consequently, a host task placed between the markers is
part of the queue elapsed time even though it is not GPU-active time. The result is always calculated as
``end - start``. If the end marker is reached before the start marker, the returned duration is negative.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>160_events.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/160_events.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
