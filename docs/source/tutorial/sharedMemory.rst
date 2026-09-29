.. _shared-memory:

Shared Memory
=============

Shared memory is a scratchpad memory accessible only by threads within a thread block in a kernel.
It is useful when several threads in the same block need to reuse the same data or communicate through a fast local data chunk.
Typical use cases are, chunked stencil kernels, block-local reductions and scans, transposes, and small reusable data sets loaded once and consumed many times.
The amount of shared memory per thread block depends on the device and is usually limited to around 64 KiB, so it is not a good choice for large data sets.
In alpaka there are three common ways to declare shared memory:

- Declare a single shared value with ``declareSharedVar()``.
- Declare fixed-size shared multidimensional array or chunk with compile-time known extents with ``declareSharedMdArray()``.
- And dynamic shared memory with ``getDynSharedMem()`` when the size is only known at launch time.

A Single Shared Value
---------------------

Not every shared-memory kernel needs a shared data chunk. Sometimes one shared scalar is enough.
The next example is a very simple form of a global reduction with atomics.
All threads within a thread block accumulate into a shared memory thread block partial result.
After all threads in the thread block have finished, a single thread accumulates the partial result into the output.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedScalarKernel
    :end-before: END-TUTORIAL-sharedScalarKernel
    :dedent:

This pattern is useful for block-local counters, flags, or partial reductions.
The important detail is that the scalar still belongs to the whole thread block, not to a single thread.

.. attention::

  Do **not** forget to store the return type explicitly as reference, in this case ``auto&``, otherwise you will get a **thread local copy** instead of the shared one.

Static Shared Memory Array
--------------------------

The next example shows a chunk-wise permutation of the indices.
For each chunk the IDs should be stored in reverse order into the output.
The frame extent from the kernel launch parameters and the chunk extents are not required to match.
The chunk extent is a ``CVec`` and therefore known at compile time, this allows its usage as extents to declare static shared memory.
Static shared memory compared to dynamic shared memory, shown in the next example,
has the benefit that the developer is not required to manage the shared memory chunk by hand and in case it is multidimensional it provides address calculation optimizations.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedKernel
    :end-before: END-TUTORIAL-sharedKernel
    :dedent:

The "reverse order" work is only there to keep the example small.
The same structure is what you would use in more realistic kernels:

- Load a small image chunk before applying a blur or stencil.
- Stage a matrix chunk before a transpose or matrix multiply step.
- Cache a short chunk of data before several neighboring threads reuse it.

Launching a Shared-Memory Kernel
--------------------------------

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-sharedLaunch
    :end-before: END-TUTORIAL-sharedLaunch
    :dedent:

Dynamic Shared Memory
---------------------

Dynamic shared memory is useful when the amount of shared memory depends on kernel launch parameters or the kernel arguments.
In alpaka it will be automatically allocated indirectly for each thread block before kernel invocation.
Again the chunk-wise index reverse example is used.
The difference to the example before is that the chunk extent is now a runtime value and to get the shared memory within the kernel ``onAcc::getDynSharedMem<T>(acc)`` is used.
You will only get a flat pointer to the allocated data without any information about how many values are valid.
The developer is responsible that the number of allocated bytes at kernel launch time and used within a kernel match.

There are two supported ways to tell alpaka how many bytes to reserve.

Dynamic Size Through a Kernel Member
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most direct option is to give the kernel object a public ``uint32_t dynSharedMemBytes`` member.
This works well when the required size is already known when the kernel object is created.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedMemberKernel
    :end-before: END-TUTORIAL-dynSharedMemberKernel
    :dedent:

When you launch that kernel, set the byte count in the kernel object itself.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedMemberLaunch
    :end-before: END-TUTORIAL-dynSharedMemberLaunch
    :dedent:

This form is simple and readable, but it is intentionally limited: the size can only depend on data you put into the kernel object.

Dynamic Size Through ``BlockDynSharedMemBytes`` Trait
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the size should depend on the executor or the kernel arguments, alpaka uses a trait specialization.
If you call a kernel with a frame specification the thread specification in the constructor of the trait will be the derived specification used to launch the kernel.
The user-defined data chunk size passed through the kernel arguments is used to calculate the required amount of shared memory to hold a single chunk per thread block.
The required data to hold a chunk is intended to be independent of the thread specification to control the amount of reused data.
If you provide neither a ``dynSharedMemBytes`` member nor a trait implementation ``alpaka::onHost::trait::BlockDynSharedMemBytes`` specialization, alpaka reserves no dynamic shared memory for that kernel.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitSpec
    :end-before: END-TUTORIAL-dynSharedTraitSpec
    :dedent:

The kernel itself still uses ``getDynSharedMem`` in the normal way.
If your kernel provides a member ``uint32_t dynSharedMemBytes`` as shown in the previous example the member variable is ignored and the trait specialization is used instead.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitKernel
    :end-before: END-TUTORIAL-dynSharedTraitKernel
    :dedent:

The difference when launching the kernel in comparison to the previous example is that the kernel is not initialized with the byte value and there is an additional chunk size argument.

  .. literalinclude:: ../../snippets/example/120_sharedMemory.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-dynSharedTraitKernelChunked
    :end-before: END-TUTORIAL-dynSharedTraitKernelChunked
    :dedent:

Practical Advice
----------------

- Shared memory is local to the thread block. Different blocks cannot see each other's shared data.
- Shared memory is not initialized automatically.
- Every thread that reads shared data written by other threads usually needs a block synchronization first.
- Reusing the same shared-memory id returns the same storage again; a different id gives you different storage.
- Use ``declareSharedVar()`` for a single shared scalar or one small fixed object.
- Use ``declareSharedMdArray()`` for multidimensional data.
- Use ``getDynSharedMem()`` when the temporary size depends on kernel arguments.
- Start with small chunks and a simple mapping before trying to micro-optimize the memory layout.

Common Mistakes
---------------

- Treating shared memory as if different blocks could see the same storage.
- Reading shared values before a required block synchronization.
- Introducing shared memory before checking that the data is reused.
- Using dynamic shared memory when a small fixed chunk would already be simpler and clearer.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>120_sharedMemory.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/120_sharedMemory.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
