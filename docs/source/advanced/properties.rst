.. _properties:

Properties
==========

Memory Properties
-----------------

The properties in this subsection are available in the namespace ``alpaka::memoryProperty``. They can be applied to all allocation calls: ``alloc``, ``allocHost``, ``allocMapped``, ``allocUnified``, ``allocLike``, ``allocHostLike``, ``allocDeferred`` and ``allocLikeDeferred``.

- ``memoryProperty::bestLatency`` This property selects memory nodes with the lowest access latency for memory pinning
- ``memoryProperty::bestBandwidth`` This property selects memory nodes with the highest memory bandwidth for memory pinning
- ``memoryProperty::locality`` This property selects the closest memory nodes for memory pinning
- ``memoryProperty::defaultProperty`` Equivalent to no property; uses the default alpaka3 behavior

Only one property of this category may be given per allocation call. If none is given, ``memoryProperty::defaultProperty`` is used.

Device Support
--------------

Not every device implements every memory property. A device declares the properties it can honour, and an allocation which requests a property the device does not implement is rejected at compile time.

- ``onHost::hasProperty(deviceOrQueue, properties...)`` returns the first property of the list which the device implements
- ``onHost::isMemoryPropertySupportedBy(property, deviceOrQueue)`` returns whether the device can honour the property
- ``onHost::supportedMemoryProperties(deviceOrQueue)`` returns a tuple with all properties the device can honour

``hasProperty`` is variadic and scans its arguments from left to right, returning the first property the device
implements. A property that belongs to a different kind of device is skipped rather than rejected, which is what keeps
a single source portable across backends:

.. code-block:: cpp

   // compile time error, a GPU does not implement a NUMA placement preference
   auto bufferA = onHost::alloc<int>(gpuDevice, extents, memoryProperty::bestBandwidth);

   // returns memoryProperty::locality on a CPU device, someGpuProperty is skipped
   auto property = onHost::hasProperty(
       cpuDevice,
       someGpuProperty,
       memoryProperty::locality,
       memoryProperty::defaultProperty);
   auto bufferB = onHost::alloc<int>(cpuDevice, extents, property);

The whole selection happens at compile time, the returned type is the selected property tag.

The fallback is never implicit. If the device implements none of the listed properties the call is a compile time
error, so that a request is not quietly turned into something else:

.. code-block:: cpp

   // compile time error, a GPU implements none of these
   auto p1 = onHost::hasProperty(gpuDevice, memoryProperty::bestBandwidth, memoryProperty::locality);
   // ok, accepting the backend default placement is written down explicitly
   auto p2 = onHost::hasProperty(gpuDevice, memoryProperty::bestBandwidth, memoryProperty::defaultProperty);

``memoryProperty::defaultProperty`` is implemented by every device, so ending the list with it is how a portable call
states that the backend default placement is acceptable.

Which Devices Implement What
----------------------------

Currently only CPU devices (``deviceKind::cpu`` and ``deviceKind::numaCpu``) implement ``bestLatency``,
``bestBandwidth`` and ``locality``, and only when alpaka is **built with hwloc**. hwloc is what carries the placement
out, so in a build without it a CPU device implements nothing but ``memoryProperty::defaultProperty`` and requesting a
placement preference is a compile time error, exactly as it is on a GPU.

This means a source which passes a placement preference *directly* is specific to an hwloc-enabled build. Routing it
through ``hasProperty`` keeps it compiling either way:

.. code-block:: cpp

   // needs a build with hwloc, compile time error without it
   auto bufferA = onHost::alloc<int>(cpuDevice, extents, memoryProperty::bestBandwidth);
   // compiles with and without hwloc
   auto bufferB = onHost::alloc<int>(
       cpuDevice, extents, onHost::hasProperty(cpuDevice, memoryProperty::bestBandwidth, memoryProperty::defaultProperty));

All other devices implement only ``memoryProperty::defaultProperty``.

Runtime Availability
--------------------

Whether a device *implements* a property is a compile time question, whether the machine the program runs on can
actually *deliver* it is not. A CPU device in an hwloc build implements ``bestBandwidth`` even on a machine whose
firmware exposes no memory attribute data, because that is only discoverable once the program runs.

When the property cannot be carried out at runtime, alpaka prints a warning on ``std::cerr`` and the allocation keeps
the default placement. This happens when:

- hwloc provides no data for the requested attribute on this machine, for example a platform without ACPI HMAT
- the OS or container refuses the memory binding (missing privileges, restricted cpuset)

The allocation always succeeds, it just may not be placed as asked. Each distinct warning is printed only once per
process. Set the environment variable ``ALPAKA_NO_WARNINGS`` to silence them.

``memoryProperty::locality`` is the most widely available preference, because hwloc derives it from the topology
itself rather than from firmware tables.

A backend declares its support by specializing ``alpaka::onHost::trait::IsMemoryPropertySupportedBy::Op``:

.. code-block:: cpp

   template<typename T_Platform>
   struct alpaka::onHost::trait::IsMemoryPropertySupportedBy::Op<
       alpaka::memoryProperty::BestBandwidth,
       MyDevice<T_Platform>> : std::true_type
   {
   };

Memory Policy List
------------------

Allocation properties are collected in an ``alpaka::onHost::MemoryPolicyList``. Properties can either be passed directly to the allocation call or bundled explicitly into a policy list, both forms are equivalent:

.. code-block:: cpp

   auto bufferA = onHost::alloc<int>(device, extents, memoryProperty::bestBandwidth);
   // is equivalent to
   auto bufferB = onHost::alloc<int>(device, extents, onHost::MemoryPolicyList{memoryProperty::bestBandwidth});

See :ref:`memory-pinning`
