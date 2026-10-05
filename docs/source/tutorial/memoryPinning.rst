.. _memory-pinning:

Memory Pinning
==============

On the host, or when CPU devices are used, users can select which kind of memory is allocated. This can be done using allocation properties to select the memory with the best latency, best bandwidth, or best locality.

Memory Allocation Property
--------------------------

- ``alpaka::memoryProperty::bestLatency`` Selects memory nodes with the best access latency. Only CPU devices with HWLOC.
- ``alpaka::memoryProperty::bestBandwidth`` Selects memory nodes with the highest memory bandwidth. Only CPU devices with HWLOC.
- ``alpaka::memoryProperty::locality`` Selects the closest memory nodes. Only CPU devices with HWLOC.
- ``alpaka::memoryProperty::defaultProperty`` Equivalent to no property; uses the default alpaka3 behavior. Supported by all devices

These properties bind the memory to the selected nodes. If the memory nodes are full, the allocation fails. They use the MPOL_BIND policy.

Only devices which implement a property accept it. Passing ``memoryProperty::bestBandwidth`` to a device which cannot honour it is a compile time error. That covers a GPU, and also a CPU device in a build without hwloc, since hwloc is what carries the placement out. To write a single source which runs on every backend and every build configuration, select the property with ``alpaka::onHost::hasProperty()``. It takes a preference list, scans it from left to right and returns the first property the device implements, so no separate code path per device is needed:

.. literalinclude:: ../../snippets/example/250_memoryProperties.cpp
  :language: cpp
  :start-after: BEGIN-TUTORIAL-allocPortable
  :end-before: END-TUTORIAL-allocPortable
  :dedent:

If the device declares the property but the machine cannot deliver it, for example because hwloc has no data for the attribute on this hardware, a warning is printed once on ``std::cerr`` and the allocation keeps the default placement. The allocation itself always succeeds in this case. Users cannot work around this issue. Set ``ALPAKA_NO_WARNINGS`` to suppress the warning.

See :ref:`properties <properties>` for how a backend declares the properties it supports.

Works with all kinds of allocations:

- Device allocation

  .. literalinclude:: ../../snippets/example/250_memoryProperties.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDev
    :end-before: END-TUTORIAL-allocBufferDev
    :dedent:

- Mapped allocation

  .. literalinclude:: ../../snippets/example/250_memoryProperties.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferMapped
    :end-before: END-TUTORIAL-allocBufferMapped
    :dedent:

- Unified Memory

  .. literalinclude:: ../../snippets/example/250_memoryProperties.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferUnified
    :end-before: END-TUTORIAL-allocBufferUnified
    :dedent:

- Alloc Like

  .. literalinclude:: ../../snippets/example/250_memoryProperties.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocLike
    :end-before: END-TUTORIAL-allocLike
    :dedent:

- Deferred Allocation

  .. literalinclude:: ../../snippets/example/250_memoryProperties.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDeferred
    :end-before: END-TUTORIAL-allocBufferDeferred
    :dedent:

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>250_memoryProperties.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/250_memoryProperties.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
