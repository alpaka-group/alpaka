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

Memory Policy List
------------------

Allocation properties are collected in an ``alpaka::onHost::MemoryPolicyList``. Properties can either be passed directly to the allocation call or bundled explicitly into a policy list, both forms are equivalent:

.. code-block:: cpp

   auto bufferA = onHost::alloc<int>(device, extents, memoryProperty::bestBandwidth);
   // is equivalent to
   auto bufferB = onHost::alloc<int>(device, extents, onHost::MemoryPolicyList{memoryProperty::bestBandwidth});

See :ref:`memory-pinning`
