Views and Subviews
==================

Buffers are objects returned by memory allocation methods. They own the data stored in memory and track the lifetime of that memory. They follow the :ref:`buffer concept <i_buffer>`.
On the other hand, :ref:`Views <i_view>` only point to memory but do not own it and can be used within a kernel in addition to on the host.
Unlike an :ref:`MdSpan <i_mdspan>`, the :ref:`Views <i_view>` contain information about which :ref:`api` was used to allocate the memory.
This information is required for certain functions, such as copying memory.

Views are typically used when:

- You already have a host container such as ``std::vector`` and want to use it with *alpaka*.
- You want to work only on parts of a buffer, for example a slice, halo region, or tile.
- You want to use the data of a buffer within a kernel without copying it.

Creating a View
---------------

You can create a non-owning view from *alpaka* buffers and then derive a subview from it.

  .. literalinclude:: ../../snippets/example/070_views.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-bufferViewCreation
    :end-before: END-TUTORIAL-bufferViewCreation
    :dedent:

Creating a view from an STL vector is supported too.
Creating a subview is the same as for buffers.

  .. literalinclude:: ../../snippets/example/070_views.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-viewCreation
    :end-before: END-TUTORIAL-viewCreation
    :dedent:

This is useful when the data already exists and you want to keep using the original storage.
For example, a stencil update often wants the interior cells only, while a boundary kernel wants a narrow halo view around the edge.

Copying Through a View
----------------------

Views work with the usual memory operations.
That means you can, for example, allocate device memory based on a view and copy only the relevant slice back.

  .. literalinclude:: ../../snippets/example/070_views.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-viewCopy
    :end-before: END-TUTORIAL-viewCopy
    :dedent:

Typical use cases include:

- Copying a subrange of a 1D vector.
- Only the inner area of the grid containing the actual data, without the halo, should be copied into a stencil code.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>070_views.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/070_views.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
