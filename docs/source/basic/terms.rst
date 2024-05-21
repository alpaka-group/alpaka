Terms
=====

This page provides an overview of the terms used in ``alpaka`` and the relationships between them.

.. image:: /images/alpaka_terms.svg

Platform
--------

- A ``platform`` contains information about the system, e.g. the available devices. 
- Depending on the platform, it also contains a runtime context.
- A ``platform`` has a least one device but it can also has many device.
- Each ``platform`` can be used with N ``accelerator``. ``platforms`` and ``accelerator`` cannot freely combined. An ``accelerator`` supports only a specific ``platform``.
 
Device
------

- A ``device`` represent a compute unit, such as a CPU or a GPU.
- Each ``device`` is bounded to a specific ``platform``.
- Each ``device`` can have N ``queues``.

Accelerator
-----------

- A ``accelerator`` is a index mapping function. It distributes the index space to the chunks. The ``accelerator`` maps a continues index space to a blocked index domain decomposition.
- It is not allowed to create an instance of an ``accelerator``.
- A ``accelerator`` is bounded to a specific ``platform``.

Queue
-----

- Stores operations which should be executed on a ``device``.
- Operations can be ``TaskKernels``, ``Events``, ``Sets`` and ``Copies``.
- Each ``queue`` is bounded to a specific ``device``.

TaskKernel
----------

- A ``TaskKernel`` contains the algorithm which should be executed on a ``device``.

Event
-----

- A ``event`` is a marker in the ``queue``.
- ``events`` can be used to describe dependencies between different ``queues``.
- A ``event`` allows to wait until a specific time point.

Set
---

- A ``Set`` set byte wise a memory to a specific value.

Copy
----

- Copies memory from memory location to another memory location.
