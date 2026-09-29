Allocate Memory
===============

Now that we know how to :ref:`get a device <device-selection>` and create :ref:`a queue <queue_creation>`, we can move on to memory allocation.
To allocate memory, you need a *device* and sometimes a *queue*.
alpaka's memory allocation methods return a *DataStorage* object that implements the ``alpaka::concepts::IBuffer`` interface.
The ``alpaka::concepts::IBuffer`` interface requires that the lifetime of the memory be tracked.
One implementation of this interface is the ``alpaka::onHost::SharedBuffer`` handle.
It behaves like a ``std::shared_ptr<>`` from the standard library.
When the last instance goes out of scope, the memory is freed.

- If a *DataStorage* handle allows it, copying a handle is a shallow copy of the handle and does not duplicate the data.
- A deep copy of the memory must be explicitly triggered using ``alpaka::onHost::memcpy()``.
- A buffer is **not** initialized with default values.
- The *extents*, which describe the number of elements per dimension, should be ``>=1``. The *extents* can have any dimensionality.
- If the extent requires the ``alpaka::concepts::VectorOrScalar`` concept, it is permissible to use a scalar instead of an alpaka vector type to allocate a one-dimensional buffer.
- The extent object also determines the internal index type used for addressing the buffer.

Choosing Extents
----------------

The extent rules above are easiest to see in code:

  .. literalinclude:: ../../snippets/example/040_memory_allocation.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferExtentForms
    :end-before: END-TUTORIAL-allocBufferExtentForms
    :dedent:

This snippet shows the three extent-related points the chapter relies on:

- a one-dimensional allocation can use a scalar extent,
- a multidimensional allocation uses an alpaka vector extent,
- and the extent's scalar type becomes the buffer's internal ``index_type``.

The following examples show how to create memory which is **only** accessible on the device.

  .. literalinclude:: ../../snippets/example/040_memory_allocation.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDev
    :end-before: END-TUTORIAL-allocBufferDev
    :dedent:

There is a type of memory called mapped memory, which is located on the CPU but is also accessible on the device.
Explicit memory copies are not required to access the memory from the device or host.
When using mapped memory, you must be careful not to access the memory of the host and the device in parallel.
Accessing this type of memory from the device is usually associated with high latency and is slow.

  .. literalinclude:: ../../snippets/example/040_memory_allocation.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferMapped
    :end-before: END-TUTORIAL-allocBufferMapped
    :dedent:

Unified memory is similar to mapped memory in that it does not require explicit memory copies.
Depending on the API used, it is located on the host or device.
It is transparently migrated page by page to the location from which it is accessed.
You should not access unified memory in parallel from the host and the device.
The first access to a memory location is often associated with high latencies, but once the page has been migrated, access is just as fast as direct access to the device memory.

  .. literalinclude:: ../../snippets/example/040_memory_allocation.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferUnified
    :end-before: END-TUTORIAL-allocBufferUnified
    :dedent:

Very often, the typical pattern for memory allocation is that you create a buffer for the host and need a second buffer for the device with the same value type and dimensions.
For this, you can use ``alpaka::onHost::allocLike(device, sourceBuffer)`` to adopt all properties except the target device.
The data in the source buffer is not copied.
This can only be done explicitly.

  .. literalinclude:: ../../snippets/example/040_memory_allocation.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocLike
    :end-before: END-TUTORIAL-allocLike
    :dedent:

Sometimes you want to allocate memory that is only used as a temporary buffer and is no longer needed after your tasks are complete.
Since memory allocations are costly, you generally avoid allocating memory, for example, in a loop.
Depending on the device or queue API, ``alpaka::onHost::allocDeferred()`` automatically uses an internal caching allocator to keep allocation as cost-effective as possible.

That kind of temporary buffer shows up naturally later for things such as scan scratch storage, intermediate image tiles, or one stage of a multi-step numerical pipeline.

  .. literalinclude:: ../../snippets/example/040_memory_allocation.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-allocBufferDeferred
    :end-before: END-TUTORIAL-allocBufferDeferred
    :dedent:

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>040_memory_allocation.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/040_memory_allocation.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
