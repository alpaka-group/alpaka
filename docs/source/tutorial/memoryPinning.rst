.. _memory-pinning:

Memory Pinning
==============

On the host, or when CPU devices are used, users can select which kind of memory is allocated. This can be done using allocation properties to select the memory with the best latency, best bandwidth, or best locality.

Memory Allocation Property
--------------------------

- ``alpaka::memoryProperty::bestLatency`` Selects memory nodes with the best access latency
- ``alpaka::memoryProperty::bestBandwidth`` Selects memory nodes with the highest memory bandwidth
- ``alpaka::memoryProperty::locality`` Selects the closest memory nodes
- ``alpaka::memoryProperty::defaultProperty`` Equivalent to no property; uses the default alpaka3 behavior

These properties bind the memory to the selected nodes. If the memory nodes are full, the allocation fails. They use the MPOL_BIND policy.

When used with other devices, there is no need for a separate code path depending on the device. These memory properties are ignored.

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
