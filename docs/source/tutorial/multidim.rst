Working With Multidimensional Kernels
=====================================

Many important examples in parallel computing are naturally multidimensional:
images, matrices, heat diffusion, cellular automata, and finite-difference stencils.
*alpaka* supports multi-dimensional kernels and memory.
That's why it's usually better to implement these types of problems using multiple dimensions rather than flattening everything into a single linear index.

Choose the Kernel Shape From the Data
-------------------------------------

If the data is naturally a matrix or image, it is convenient to use two-dimensional extents and :ref:`frames <frame>`.
This avoids hand-written index decoding and makes boundary conditions easier to read.

  .. literalinclude:: ../../snippets/example/080_multidim.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimFrameSpec
    :end-before: END-TUTORIAL-multidimFrameSpec
    :dedent:

We recommend that the frame shape should follow the logical shape of the work.
Typical use cases for different dimensions are:

- 1D frames for flat vectors and simple reductions.
- 2D frames for images, matrices, and most stencil codes.
- 3D frames for volumetric problems, such as the position of particles in space.
- ND frames for algorithms that have more than three natural dimensions, such as position coordinates and discretized time, e.g., to build a layered detector within our particle simulation.

Keep in mind that the rightmost index, usually ``x``, is the fastest varying dimension in *alpaka* buffers.
This is important if you are using hand written ``for``-loops to iterate over the data instead of ``makeIdxMap``.

A Small 2D Stencil Example
--------------------------

The following kernel performs one five-point average step on a small 2D grid [#f1]_.
This introduces three important ideas at once:

- Iterating over multidimensional buffers.
- Handling boundaries explicitly.
- Reading neighboring cells by using relative offsets.

  .. literalinclude:: ../../snippets/example/080_multidim.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimKernelStructure
    :end-before: END-TUTORIAL-multidimKernelStructure
    :dedent:

The structure is still the same as in the one-dimensional tutorial:

- Ask the output buffer for its extents.
- Build ``IdxRange{extents}`` to describe the full valid multidimensional index domain.
- Map worker threads via ``makeIdxMap`` to the range.
- Guard the boundary cells, to avoid out-of-bounds accesses.
- The element itself is updated depending on the neighbor data and maybe on the element itself. Indices are calculated relative to the index of the own output coordinate.
- Update neighbor locations by adding or subtracting direction vectors from the current ``Vec`` index.

This is the natural alpaka style for stencil code.
The project examples, such as the heat-equation stencil, operate on the multidimensional index directly and move to
neighbors with vector offsets instead of splitting ``x`` and ``y`` into separate scalars and rebuilding indices.

At the beginning, the range check of data access should be done within the kernel.
While this approach does not offer the best performance, it is easier to implement correctly.
Subsequently, the range check should be moved from the kernel to before the kernel launch, and the kernel should operate on the assumption that the range is correct in order to improve performance.

If you want to optimize the boundary handling later, you can write a separate kernel for the border and launch it with a different frame shape that only covers the halo/guard region.

.. [#f1] https://en.wikipedia.org/wiki/Iterative_Stencil_Loops#Stencils

Launching the 2D Kernel
-----------------------

The host-side launch is unchanged except that the problem extents are vectors now, and the chosen ``FrameSpec`` is
2-dimensional as well.

  .. literalinclude:: ../../snippets/example/080_multidim.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-multidimKernelLaunch
    :end-before: END-TUTORIAL-multidimKernelLaunch
    :dedent:

What Users Usually Need To Know Early
-------------------------------------

The following habits are worth learning from the start:

- Keep boundary handling explicit. It is fine to have branching for the border in the first implementation.
- Use multidimensional buffers when the algorithm is multidimensional.
- Use temporary variables to avoid reading from and writing to memory multiple times. The variable name provides a meaningful name for read and write operations, making it easier to track when data is being read and written.
- Start with a clear kernel and a small test case before trying to optimize shared memory use or tuning the ``FrameSpec``

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>080_multidim.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/080_multidim.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
