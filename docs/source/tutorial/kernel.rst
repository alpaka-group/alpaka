.. _tutorial-kernel:

Kernel
======

After selecting a device, creating a queue, and allocating memory, the next step is to launch work on the device.
In *alpaka*, :ref:`kernels <kernel>` are implemented as function objects, known as *functors*.
A *functor* is a C++ ``class`` or ``struct`` that contains at least the member function ``void operator() const``, which implements the algorithm to be executed on the :ref:`device`.
A *functor* can be extended with additional functionality and is limited only by the requirement that it must be `trivially copyable <https://en.cppreference.com/cpp/language/classes#Trivially_copyable_class>`__.
In this tutorial, we will focus on a very simple :ref:`kernel`.

The :ref:`kernel` is implemented using the same principle as CUDA, HIP, or SYCL.
The function specifies which part of a problem a single hardware thread processes based on its ID.
To execute the algorithm, the :ref:`kernel` is launched in parallel.
Therefore, the same algorithm is executed n times, each time with a different thread ID.

A :ref:`FrameSpec <frame>` is used to describe the parallelism of a :ref:`kernel` launch.

Writing the Kernel
------------------

An *alpaka* :ref:`kernel` must meet the following requirements:

- The kernel must be a function object. Therefore, it must implement the function call operator with the following signature: ``ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto acc, ...) const`` or ``constexpr void operator()(onAcc::concepts::Acc auto acc, ...) const``.
- The *functor* must be `trivially copyable <https://en.cppreference.com/cpp/language/classes#Trivially_copyable_class>`__.
- Arguments of the functor must be trivially copyable and captured by value or const reference.
- The first argument is the const reference to the accelerator handle ``acc``, which follows the concepts ``onAcc::concepts::Acc``.
- The :doc:`Device Memory <memoryAllocation>` handles are passed via the arguments [#f1]_. Typically, output buffers use ``IMdSpan`` and input buffers use ``IDataSource``.

.. [#f1] We need to pass a handle instead of copying the memory directly, because the kernel must be trivially copyable. The simplest handle is a raw pointer. We strongly advise against using raw pointers, as they have many drawbacks and are a major source of errors.

.. literalinclude:: ../../snippets/example/050_kernel.cpp
  :language: cpp
  :start-after: BEGIN-TUTORIAL-kernelStructure
  :end-before: END-TUTORIAL-kernelStructure
  :dedent:

*alpaka* offers many different ways to write a :ref:`kernel`, but it also provides several functions to help you create a kernel easily and securely.
``onAcc::makeIdxMap`` is the most important helper function for writing a kernel.
It distributes a range of indices among the workers.
The second argument defines which worker group the executor threads belong to.

For the simplest kernel, we use the worker group ``onAcc::worker::threadsInGrid``, which means that the index range is distributed across all global threads [#f2]_.
In the example, the range is the extents of ``out``.
We access all elements of ``out`` in parallel and write the sum of ``lhs`` and ``rhs`` to data position ``i``.
The parallelism is defined by the :ref:`FrameSpec <frame>` when the :ref:`kernel` starts.
Within the :ref:`kernel`, we work with the parallelism defined by the :ref:`ThreadSpec <thread_spec>`, which is derived from the :ref:`FrameSpec <frame>` and the used queue.
The :ref:`tutorial_launch_kernel` section explains how this derivation works.

The function call ``onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()})`` provides us with certain guarantees:

- All elements in ``out`` are processed by the worker group independently of the parallelism specified via :ref:`FrameSpec <frame>` [#f3]_.
- This ensures that no out-of-range memory access happens.
- Indices provided by ``onAcc::makeIdxMap`` for each worker group are optimized based on the used :ref:`device` of the :ref:`queue` and the :ref:`executor`.
  They can be used for a one-to-one mapping to memory.
  Resulting in a contiguous, chunked access pattern on the CPU and a strided access pattern on GPU-like devices.

``onAcc::makeIdxMap`` offers many more features needed for performance optimization.
Once you've finished the tutorial, check out the :doc:`kernelParallelism` section to discover the full potential of this function.

.. [#f2] The total number of threads is calculated by multiplying the number of threads by the number of blocks in a :ref:`thread_spec`.
.. [#f3] It is also possible to configure a :ref:`FrameSpec <frame>` with the values `{1,1}`, which means that all elements in ``onAcc::makeIdxMap`` are processed sequentially.

.. _tutorial_launch_kernel:

Launching the Kernel
--------------------

On the host side, we first need to create a :ref:`FrameSpec <frame>`.
A :ref:`FrameSpec <frame>` describes the maximum possible parallelism.
A :ref:`FrameSpec <frame>` contains the number of :ref:`Frames <frame>` and the :ref:`Frame Extents <frame>`.
When the :ref:`kernel` starts, a :ref:`thread_spec` to execute the kernel is derived from the :ref:`FrameSpec <frame>`, the :ref:`executor <executor>` and :ref:`queue <queue>` properties.
Therefore, the number of threads per thread block is derived from :ref:`Frame Extents <frame>`, and the number of thread blocks is derived from :ref:`Frame count <frame>`.
The numbers from the :ref:`Frame Extents <frame>` can be directly transferred to the :ref:`thread_spec`, but the number of blocks and/or threads may also be smaller than in the :ref:`FrameSpec <frame>`.

Let's take, for example, a :ref:`FrameSpec <frame>` with the dimensions {10, 32} (number of Frames, Frame extents):

- On a GPU, :ref:`thread_spec` could take the value ``{10, 32}`` (blocks, threads), since we can launch 10 blocks and each GPU core has 32 threads.
- On a CPU, the :ref:`thread_spec` could be ``{10, 1}``  (blocks, threads), since we can run 10 blocks on 10 or fewer cores, but each core can have only one thread.

``onAcc::makeIdxMap`` allows the use of any number of *blocks* and *threads*.
Therefore, any :ref:`FrameSpec <frame>` will work.
The size of the :ref:`FrameSpec <frame>` effects performance but *not* the correctness.
To begin with, you should use the ``onHost::getFrameSpec`` function to create a :ref:`FrameSpec <frame>`.
This function assumes that you are using the ``onAcc::makeIdxMap`` function in your :ref:`kernel` and returns a well-functioning :ref:`FrameSpec <frame>` depending on the :ref:`device` and :ref:`executor`.

.. literalinclude:: ../../snippets/example/050_kernel.cpp
  :language: cpp
  :start-after: BEGIN-TUTORIAL-kernelLaunch
  :end-before: END-TUTORIAL-kernelLaunch
  :dedent:

In the advanced area, section :doc:`kernelParallelism`, you'll learn how ``onAcc::makeIdxMap`` works in detail.
Once you understand this, you can manually set a :ref:`FrameSpec <frame>` that might work better than the :ref:`FrameSpec <frame>` returned by ``onHost::getFrameSpec``.

Typical Beginner Mistakes
-------------------------

- Forgetting to copy the inputs to the device before enqueueing the kernel.
- Forgetting to copy the result back to the host after the kernel.
- Forgetting to wait before reading host-side results from a non-blocking queue.
- Choosing a one-dimensional frame for naturally multidimensional code and then reimplementing manual index arithmetic in the kernel.
- Calculating your own thread ID and using it to access the data, rather than accessing the data via ``onAcc::makeIdxMap``.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>050_kernel.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/050_kernel.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
