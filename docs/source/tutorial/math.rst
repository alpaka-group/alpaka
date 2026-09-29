Math Functions
==============

Inside kernels, prefer ``alpaka::math`` over calling backend-specific math APIs or C++ `std` math functions directly.
*alpaka* provides a wide range of mathematical functions in the `alpaka::math <../doxygen/namespacealpaka_1_1math.html>`_ namespace.
These enable a portable codebase while delivering the best possible performance for the data types used.
To this end, *alpaka* uses the native math functions of the :ref:`api` (e.g., CUDA built-in math functions) with which a kernel is executed, whenever available.
*alpaka* math functions can also be used in host code outside of kernels.

Element-wise Math
-----------------

The example is similar to the vector addition: iterate over the data with ``makeIdxMap`` and call math functions on each element.

  .. literalinclude:: ../../snippets/example/140_math.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-mathKernel
    :end-before: END-TUTORIAL-mathKernel
    :dedent:

Distance-like Computations
--------------------------

Reciprocal square root is another common operation in physics, graphics, and geometry kernels.

  .. literalinclude:: ../../snippets/example/140_math.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-rsqrtKernel
    :end-before: END-TUTORIAL-rsqrtKernel
    :dedent:

Available Function Families
---------------------------

.. include:: ../../build/math_function_families.rst

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>140_math.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/140_math.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
