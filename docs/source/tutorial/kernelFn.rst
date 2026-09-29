.. _tutorial-fnkernel:

Kernel Function
===============

In the :ref:`kernel tutorial <tutorial-kernel>` you learned that an *alpaka* kernel is represented by a functor.
Internally, a kernel launch always invokes a functor, but *alpaka* also provides a convenient way to implement kernels as free functions.
This approach offers a simple syntax for specializing a kernel implementation for a specific API or device kind while still using the same kernel interface from the application code.

This functionality is provided by *alpaka*'s generic function interface.

Writing a Kernel as a Function
------------------------------

This example uses the simple one-dimensional vector-add kernel from the :ref:`kernel tutorial <tutorial-kernel>`.

Instead of defining a functor with ``operator()``, you implement a function named ``alpakaFnDispatch``.
The first argument must be an *alpaka* function symbol or a function-symbol specialization, which allows *alpaka* to select the correct implementation at compile time.
Function symbols and all corresponding ``alpakaFnDispatch`` overloads must be placed inside a namespace.
*alpaka* relies on `Argument-Dependent Lookup (ADL) <https://de.wikipedia.org/wiki/Argument_dependent_name_lookup>`_ to find matching overloads, and functions located in the global namespace cannot be found reliably.

The following example defines a generic vector-add kernel implementation that can run on any supported backend.

.. literalinclude:: ../../snippets/example/230_kernelFn.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-kernelFnKernel
   :end-before: END-TUTORIAL-kernelFnKernel
   :dedent:

Launching a kernel function is identical to launching a kernel functor.
The function symbol behaves like a stateless functor with a default constructor and can therefore be passed directly to a ``KernelBundle``.

.. literalinclude:: ../../snippets/example/230_kernelFn.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-kernelFnLaunch
   :end-before: END-TUTORIAL-kernelFnLaunch
   :dedent:

Specializing a Kernel Function for CUDA
---------------------------------------

In some situations you may want to provide a backend-specific implementation.
Typical reasons include:

* using native backend features that are not available through the portable abstraction
* achieving maximum performance for a specific platform
* gradually porting existing CUDA code to *alpaka*

This can be achieved by overloading ``alpakaFnDispatch`` for a specific API and device kind.
When a kernel is launched on a matching backend, *alpaka* automatically selects the specialized overload.

The example below provides a CUDA-specific implementation using native CUDA thread and grid indices.

Because this implementation depends on CUDA-specific language features such as ``blockIdx`` and ``threadIdx``, it must be guarded with a pre-compiler macro ``ALPAKA_LANG_CUDA``.
Without this guard, compilation would fail on systems where CUDA support is not enabled.

.. literalinclude:: ../../snippets/example/230_kernelFn.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-kernelFnCudaKernel
   :end-before: END-TUTORIAL-kernelFnCudaKernel
   :dedent:

Overload Resolution
-------------------

When a kernel is launched, *alpaka* selects the most specialized matching ``alpakaFnDispatch`` overload.

For example:

* ``alpakaFnDispatch(VectorAdd, ...)`` provides a generic implementation.
* ``alpakaFnDispatch(VectorAdd::Spec<api::Cuda, T_DeviceKind>, ...)`` provides a CUDA-specific implementation.
* ``alpakaFnDispatch(VectorAdd::Spec<api::OneApi, deviceKind::IntelGpu>, ...)`` provides an implementation for the Intel GPU accessed via OneApi. (not used in this example)

If the kernel is executed on a queue for a CUDA device, the CUDA specialization is selected.
For all other backends, the generic implementation is used automatically.
This allows application code to remain unchanged while backend-specific optimizations are added where needed.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>230_kernelFn.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/230_kernelFn.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
