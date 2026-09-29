Memory Fences
=============

``onAcc::memFence`` is a visibility and ordering primitive inside kernels.
It is not a barrier, therefore it does not wait for other threads to reach the same point.
Instead, it tells the backend how data writes before the fence must become visible relative to data reads and writes after the fence.

With a `scope <../doxygen/namespacealpaka_1_1onAcc_1_1scope.html>`_ you define between which thread hierarchy levels the visibility guarantee applies.

- ``onAcc::scope::block`` for communication inside one thread block,
- ``onAcc::scope::device`` for communication across blocks on the same device.

With an `order <../doxygen/namespacealpaka_1_1onAcc_1_1order.html>`_ you specify the ordering guarantee of the memory operations.

Block-Scope Ordering
--------------------

The first example manages thread block shared data ordering without atomics.
One thread publishes two values in shared memory.
The fence guarantees that the write to ``shared[0]`` becomes visible before the later write to ``shared[1]`` is observed as published.
On a device with API ``host`` and device kind ``cpu`` there is no possibility that wrong values are read because the number of threads within a thread block is always one.
On parallel devices, e.g. a GPU, the ordering guarantee is required.

  .. literalinclude:: ../../snippets/example/180_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceBlockKernel
    :end-before: END-TUTORIAL-memFenceBlockKernel
    :dedent:

The kernel start with a frame spec and the initialization of the success flag are shown below.

  .. literalinclude:: ../../snippets/example/180_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceBlockLaunch
    :end-before: END-TUTORIAL-memFenceBlockLaunch
    :dedent:

Device-Scope Publication
------------------------

The second example shows the classic producer/consumer publication pattern in global memory.
The producer writes the payload, issues a release fence, and only then atomically sets a ready flag.
The consumer spins on the atomic ready flag, issues an acquire fence, and then reads the payload.
This example intentionally uses ``ThreadSpec`` instead of ``FrameSpec`` because the algorithm needs an exact guarantee
about how many thread blocks and threads are launched.

  .. literalinclude:: ../../snippets/example/180_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceDeviceKernel
    :end-before: END-TUTORIAL-memFenceDeviceKernel
    :dedent:

  .. literalinclude:: ../../snippets/example/180_memFence.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-memFenceDeviceLaunch
    :end-before: END-TUTORIAL-memFenceDeviceLaunch
    :dedent:

Used pattern:

- producer: write data, ``memFence(..., scope::device, order::release)``, then atomically publish the flag
- consumer: atomically observe the flag, ``memFence(..., scope::device, order::acquire)``, then read the data

Practical Advice
----------------

- A fence orders memory operations; it does not make conflicting non-atomic writes safe.
- Keep the publication protocol simple: payload first, fence second, atomic flag update last.
- For best performance use ``scope::block`` over ``scope::device`` when block-local visibility is enough.
- Use the weakest memory order that expresses the algorithm clearly. ``release`` / ``acquire`` is often the right pair for producer/consumer publication.
- The meaning stays the same across backends, but the runtime cost can differ.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>180_memFence.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/180_memFence.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
