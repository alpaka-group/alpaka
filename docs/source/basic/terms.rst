Terms
=====

This page provides an overview of the terms used in ``alpaka`` and the relationships between them.

Platform
--------

* A ``platform`` contains information about the system, e.g. the available devices. 
* Depending on the platform, it also contains a runtime context.
* A ``platform`` can be shared by many devices.

Device
------

* A ``device`` represent a compute unit, such as a CPU or a GPU.
* Each ``device`` is bounded to a specific ``platform``.
* Each ``device`` can be used by many specific ``accelerators``.

Work division
-------------

* Describes the domain decomposition of a contiguous N-dimensional index domain in ``blocks``, ``threads`` and ``elements``. A ``block`` contains one or more ``threads`` and a ``thread`` process one or more ``elements``.
* A ``work division`` has limitations depending on the ``kernel`` function and ``accelerator``.

Accelerator
-----------

* Describes "how" a kernel work division is mapped to device threads.
    * N-dimensional work divisions (1D, 2D, 3D) are supported.
    * Holds implementations of shared memory, atomic operations, math operations etc.
* ``Accelerators`` are instantiated only when a kernel is executed, and can only be accessed in device code.
    * Each device function can (should) be templated on the accelerator type, and take an accelerator as its first argument.
    * The accelerator object can be used to extract the ``work division`` and indices of the current block and thread.
    * The accelerator type can be used to implement per-accelerator behaviours.
* An ``accelerator`` is bounded to a specific ``platform``.

Queue
-----

* Stores tasks which should be executed on a ``device``.
* Operations can be ``TaskKernels``, ``HostTasks``, ``Events``, ``Sets`` and ``Copies``.
* A Queue can be ``Blocking`` (host thread is waiting for finishing the API call) or ``NonBlocking`` (host thread continues after calling the API independent if the call finished or not).
* All operations in a queue will be executed sequential in FIFO order.
* Operations in different queues can run in parallel even on blocking queues.
* ``wait()`` can be executed for queues to block the caller host thread until all previous enqueued work is finished.
* Each ``queue`` is bounded to a specific ``device``.

Task
----

* A ``TaskKernel`` contains the algorithm which should be executed on a ``device``.
* A ``HostTask`` is a functor without ``acc`` argument, which can be enqueued and is always executed on the host device. 

Event
-----

* A ``event`` is a marker in a ``queue``.
* ``events`` can be used to describe dependencies between different ``queues``.
* A ``event`` allows to wait until all previous enqueued work in a queue has finished.

Set
---

* A ``Set`` sets a memory region to a specific value byte-wise.

Copy
----

* Deep memory copy from one memory to another memory location.
