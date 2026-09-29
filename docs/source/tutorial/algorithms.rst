Host-Side Algorithms
====================

*alpaka* provides host-side algorithms that execute on the selected :ref:`backend <backend>` through an *alpaka* queue and :ref:`executor <executor>`.
They operate on *alpaka* :ref:`basic-data-storage`.
All algorithms except ``scan`` can operate on n-dimensional data.
They are implemented to use `SIMD data packs <https://en.wikipedia.org/wiki/Single_instruction,_multiple_data>`_ to improve the memory bandwidth and compute utilization.
SIMD is often explicitly exposed to provide you the full control and optimization potential.

General
-------

A host-side algorithm consists of the following components:

- **Algorithm**: The function itself determines how the input data or configuration is processed.
- **Queue**: An *alpaka* queue for organizing the execution order along with other operations.
- **Executor**: Determines the level of parallelism used to execute the algorithm. The :ref:`executor` is optional. If not specified, *alpaka* is using a fitting executor depending on the API of the backend.
- **Output**: A :ref:`basic-data-storage` that contains the result of the algorithm’s execution.
- **Input(s) or configuration**: Depending on the algorithm, one or more input :ref:`basic-data-storage` are supported, or an initial configuration is required, as is the case with the :ref:`algo_iota` algorithm, for example.
- **Functor**: The functor defines how each element of the input is processed. The signature of the required functor varies depending on the algorithm and functor type. The :ref:`algo_transform` illustrates the different functors. Some algorithms, such as :ref:`algo_iota`, do not require a functor because they use a configuration.

.. _algo_transform:

Transform (ScalarFunc)
----------------------

``onHost::transform`` is the host-side algorithm equivalent of an element-wise kernel.
It applies a functor to one or more inputs and writes the result into an output buffer.
The data type of the input and output buffers can differ.

    .. literalinclude:: ../../snippets/example/130_algorithms.cpp
        :language: cpp
        :start-after: BEGIN-TUTORIAL-transformFunctor
        :end-before: END-TUTORIAL-transformFunctor
        :dedent:

    .. literalinclude:: ../../snippets/example/130_algorithms.cpp
        :language: cpp
        :start-after: BEGIN-TUTORIAL-transformCall
        :end-before: END-TUTORIAL-transformCall
        :dedent:

Functor Types
~~~~~~~~~~~~~

The algorithm determines how many input arguments are available in the kernel.
The data type is defined by the :ref:`basic-data-storage` for input and output.
The function type defines the wrapper type around the data types of the kernel arguments.

The default functor provides SIMD support.
The sample code above already modifies this functor.
Instead of a SIMD type, we use the scalar types ``int`` and ``float``.
This functor is typically used, if operations are used that do not work on SIMD types (e.g. ``math::min``, ``math::max``, ...).
To use scalar types instead of SIMD types, the functor object is wrapped in the ``ScalarFunc`` object when the ``onHost::transform`` algorithm is called.
The next section shows the default functor with SIMD support.

Default Functor (SIMD)
``````````````````````

The functor takes SIMD packs as input and returns SIMD packs.
Although the operations are still written as if they were working with scalar types, they are applied in parallel to many elements.

    .. literalinclude:: ../../snippets/example/130_algorithms.cpp
        :language: cpp
        :start-after: BEGIN-TUTORIAL-transformFunctorDefaultFunc
        :end-before: END-TUTORIAL-transformFunctorDefaultFunc
        :dedent:

The ``onHost::transform`` algorithm is executed without a wrapper around the functor.

    .. literalinclude:: ../../snippets/example/130_algorithms.cpp
        :language: cpp
        :start-after: BEGIN-TUTORIAL-transformCallDefaultFunc
        :end-before: END-TUTORIAL-transformCallDefaultFunc
        :dedent:

StencilFunc
```````````

The ``StencilFunc`` functor accepts ``SimdPtr`` as arguments.
A ``SimdPtr`` supports ``operator[]`` for relative indexing, enabling stencil operations.
The dimension of the ``SimdPtr`` arguments is the same as the in- and output :ref:`basic-data-storage` object.

The following example is a 2D stencil code.

    .. literalinclude:: ../../snippets/example/130_algorithms.cpp
        :language: cpp
        :start-after: BEGIN-TUTORIAL-transformFunctorStencilFunc
        :end-before: END-TUTORIAL-transformFunctorStencilFunc
        :dedent:

You must ensure memory accesses stay in bounds. Typically a sub-view of the in- and output buffer is created that excludes the halo.

    .. literalinclude:: ../../snippets/example/130_algorithms.cpp
        :language: cpp
        :start-after: BEGIN-TUTORIAL-transformCallStencilFunc
        :end-before: END-TUTORIAL-transformCallStencilFunc
        :dedent:

.. _algo_iota:

Iota
----

``onHost::iota`` fills one or more output buffers with a linear sequence of scalar indices.
For multidimensional buffers, the linear value increases fastest in the last dimension.
In this example we start the enumeration with 10.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-iota
    :end-before: END-TUTORIAL-iota
    :dedent:

Reduction
---------

Reduction writes its result into the first element of the output buffer.
The neutral element of the operation binary reduce functor must be provided explicitly.
This allows keeping the data in device memory without going at any time to host memory.
That is different compared to `std::reduce <https://en.cppreference.com/cpp/algorithm/reduce>`_.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-reduce
    :end-before: END-TUTORIAL-reduce
    :dedent:


Transform-Reduce
----------------

``transformReduce`` combines a data transformation step with a reduction step.
It is used for dot products, weighted sums, norms, and many "compute a value per element and then accumulate it" patterns.
The first functor is the binary reduction operator. The second one is the element-wise transform and must accept as many arguments as there are input :ref:`basic-data-storage` objects passed to ``onHost::transformReduce``.
Compared to ``std::transform_reduce`` the function supports any amount of input :ref:`basic-data-storage` objects.
The following example iterates through ``inputBuffer1`` and ``inputBuffer2``, multiplies ``inputBuffer1[i]`` by ``inputBuffer2[i]``, and accumulates the products.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformReduceFunctor
    :end-before: END-TUTORIAL-transformReduceFunctor
    :dedent:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformReduceCall
    :end-before: END-TUTORIAL-transformReduceCall
    :dedent:

Writing to input arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both ``transform`` and ``transformReduce`` can write to their input arguments when the functor is wrapped in ``StencilFunc``.
The ``SimdPtr`` passed to a stencil functor supports assignment (``operator=``), allowing in-place modification of the input data.
This is useful for algorithms that need to update their inputs as they process them.
However, the user is responsible for ensuring no data races occur — the algorithm does not enforce ordering between reads and writes to the same element.

Concurrent
----------

``onHost::concurrent`` executes an n-ary functor on each element of all buffers.
There is no restriction on whether a buffer is used for read operations, write operations, or both.
This lets you implement a transform with a free number of outputs, or fuse multiple reads and writes into a single kernel launch.
The functor receives ``SimdPtr`` arguments, so stencil-style indexing works without wrapping in ``StencilFunc``.
You can also pass generators as inputs, see the :ref:`generator section <algo_generators>`.

The following example uses three buffers.
The first buffer is used for both reading and writing.
The second buffer is read-only, and the third buffer is used only for writing (but can also be read).

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-concurrentFunctor
    :end-before: END-TUTORIAL-concurrentFunctor
    :dedent:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-concurrentCall
    :end-before: END-TUTORIAL-concurrentCall
    :dedent:

**Note:** The template parameter ``T_DataType`` is an optimization parameter for SIMD support.
If all buffers have the same data type, use that data type.
Otherwise, you'll need to run tests and take measurements to achieve the best performance.

Scan
----

Unlike the other algorithms in this chapter, the current scan implementation is restricted to one-dimensional data.
That fits common `prefix-sum <https://en.wikipedia.org/wiki/Prefix_sum>`_ use cases such as offsets, compaction maps, and cumulative counters, where the logical input is already a linear sequence.
This example uses an explicit temporary storage where the size is provided via ``getScanBufferSize``, this can be used to optimize the performance in cases where scan is called multiple times.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-scan
    :end-before: END-TUTORIAL-scan
    :dedent:

*alpaka* provides the following scan variants:

- ``inclusiveScan`` and ``exclusiveScan``: `Explanation on Wikipedia <https://en.wikipedia.org/wiki/Prefix_sum#Inclusive_and_exclusive_scans>`_
- ``inclusiveScanInPlace`` and ``exclusiveScanInPlace``: same as ``inclusiveScan`` and ``exclusiveScan``, but the result is written to the input buffer instead of an extra output buffer

.. _algo_generators:

Generators Instead of Input Buffers
-----------------------------------

Several *alpaka* algorithms also accept generators as inputs.
That is useful when one input is synthetic, such as a linear index, and you do not want to materialize another buffer that consumes memory just to hold it.
``LinearizedIdxGenerator`` generates scalar indices from n-dimensional indices and behaves like an `IDataSource <../doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IDataSource.html>`_.
It behaves like a virtual buffer whose value at each position is the corresponding linear index.
In this example, we add a value from one input buffer to the generated index.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-generatorFunctor
    :end-before: END-TUTORIAL-generatorFunctor
    :dedent:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-generatorCall
    :end-before: END-TUTORIAL-generatorCall
    :dedent:

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>130_algorithms.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/130_algorithms.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
