Library Interface
=================

As described in the chapter about the :doc:`Abstraction </basic/abstraction>`, the general design of the library is very similar to *CUDA* and *OpenCL* but extends both by some points, while not requiring any language extensions.
General interface design as well as interface implementation decisions differentiating *alpaka* from those libraries are described in the Rationale section.
It uses C++ because it is one of the most performant languages available on nearly all systems.
Furthermore, C++20 allows to describe the concepts in a very abstract way that is not possible with many other languages.
The *alpaka* library extensively makes use of advanced functional C++ template meta-programming techniques.
The Implementation Details section discusses the C++ library and the way it provides extensibility and optimizability.

Structure
---------

The *alpaka* library allows offloading of computations from the host execution domain to the accelerator execution domain, whereby they are allowed to be identical.

In the abstraction hierarchy the library code is interleaved with user supplied code as is depicted in the following figure.

.. image:: /images/execution_domain.png
   :alt: Execution Domains

User code invokes library functions, which in turn execute the user provided thread function (kernel) in parallel on the accelerator.
The kernel in turn calls library functions when accessing accelerator properties and methods.
Additionally, the user can enhance or optimize the library implementations by extending or replacing specific parts.

The *alpaka* abstraction itself only defines requirements a type has to fulfill to be usable with the template functions the library provides.
These type constraints are called concepts in C++.

*A concept is a set of requirements consisting of valid expressions, associated types, invariants, and complexity guarantees.
A type that satisfies the requirements is said to model the concept.
A concept can extend the requirements of another concept, which is called refinement.* `BoostConcepts <https://www.boost.org/community/generic_programming.html>`_

Concepts allow to safely define polymorphic algorithms that work with objects of many different types.

The *alpaka* library implements a stack of concepts and their interactions modeling the abstraction defined in the previous chapter.
Furthermore, default implementations for various devices and accelerators modeling those are included in the library.
The interaction of the main user facing concepts can be seen in the following figure.

.. image:: /images/structure_assoc.png
   :alt: user / alpaka code interaction


For each type of ``Device`` there is a ``DeviceSelector`` for enumerating the available ``Device``s.
A ``Device`` is the requirement for creating ``Queues`` and ``Events`` as it is for allocating ``SharedBuffers`` on the respective ``Device``.
``SharedBuffers`` can be copied, their memory can be byte-wise set or filled element-wise.
The location of the ``SharedBuffer`` data can be on the host, on the host but mapped into the device address space, directly on the device or even in a unified memory space shared between host and device.
Copying, setting or filling a view requires the corresponding ``Copy``, ``Set`` or ``Fill`` tasks to be enqueued into the ``Queue``.
An ``Event`` can be enqueued into a ``Queue`` and its completion state can be queried by the user.
It is possible to wait for (synchronize with) a single ``Event``, a ``Queue`` or a whole ``Device``.
An ``Executor`` can be enqueued into a ``Queue`` and will execute the ``Kernel`` (after all previous tasks in the queue have been completed).
The ``Kernel`` in turn has access to the ``Accelerator`` it is running on.
The ``Accelerator`` provides the ``Kernel`` with its current index in the block or grid, their extents or other data as well as it allows to allocate shared memory, execute atomic operations and many more.


Interface Usage
---------------

Accelerator Functions
`````````````````````

Functions that should be executable on an accelerator have to be annotated with the execution domain (one of ``ALPAKA_FN_HOST``, ``ALPAKA_FN_ACC`` and ``ALPAKA_FN_HOST_ACC``).
They most probably also require access to the accelerator data and methods, such as indices and extents as well as functions to allocate shared memory and to synchronize all threads within a block.
Therefore the accelerator has to be passed in as a templated constant reference parameter as can be seen in the following code snippet.

.. code-block:: cpp

   ALPAKA_FN_ACC auto doSomethingOnAccelerator(
       alpaka::onAcc::concepts::Acc auto const & acc/*,
       ...*/)                  // Arbitrary number of parameters
   -> int                      // Arbitrary return type
   {
       //...
   }


Kernel Definition
`````````````````

A kernel is a special function object which has to conform to the following requirements:

* it has to fulfill the ``std::is_trivially_copyable`` trait (has to be copyable via memcpy)
* the ``operator()`` is the kernel entry point
  * it has to be an accelerator executable function
  * it has to return ``void``
  * its first argument has to be the accelerator (templated for arbitrary accelerator back-ends)
  * all other arguments must fulfill ``std::is_trivially_copyable``

The following code snippet shows a basic example of a kernel function object.

.. code-block:: cpp

   struct MyKernel
   {
       ALPAKA_FN_ACC            // Macro marking the function to be executable on all accelerators.
       auto operator()(         // The function / kernel to execute.
           alpaka::onAcc::concepts::Acc auto const & acc/*,  // The specific accelerator implementation.
           ...*/) const         // Must be 'const'.
       -> void
       {
           //...
       }
       // Class can have members but has to be std::is_trivially_copyable.
       // Classes must not have pointers or references to host memory!
   };

The kernel function object is shared across all threads in all blocks.
Due to the block execution order being undefined, there is no safe and consistent way of altering state that is stored inside of the function object.
Therefore, the ``operator()`` of the kernel function object has to be ``const`` and is not allowed to modify any of the object members.

Kernels can also be defined via lambda expressions.

.. code-block:: cpp

   auto kernel = [] ALPAKA_FN_ACC (alpaka::onAcc::concepts::Acc auto const & acc /* , ... */) -> void {
	// ...
   }

.. attention::
   NVIDIA's ``nvcc`` compiler does not support generic lambdas which are marked with `__device__`, which is what `ALPAKA_FN_ACC` expands to (among others) when the CUDA backend is active.
   Therefore, a workaround is required. The type of the ``acc`` must be defined outside the lambda.
