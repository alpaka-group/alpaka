.. _code-example:

Code Example
============

Play online with Compiler Explorer
++++++++++++++++++++++++++++++++++

The following example can be executed online using `Godbolt Compiler Explorer <https://godbolt.org/z/K8q5q66Mn>`__. [#f1]_

.. literalinclude:: ../../snippets/example/000_elementWiseMultiplication.cpp
   :language: cpp
   :start-after: BEGIN-EXAMPLE-elementWiseMultiplication
   :end-before: END-EXAMPLE-elementWiseMultiplication
   :dedent:

.. note::

   Do not forget to set the compiler flags to compile with C++20 and optimization if you would like to inspect the assembler, e.g. ``-std=c++20 -O3``.
   If you would like to see vectorization you should set the corresponding flags, e.g. for **avx2** ``-march=haswell``.

.. [#f1] In pull requests, to test the modified code in Godbolt Compiler Explorer, replace the include with `the link to this file <../alpaka.hpp>`_.

Use alpaka in your project
++++++++++++++++++++++++++

The following example shows a small vector add example written with alpaka that can be run on different processors.

.. literalinclude:: ../../../example/vectorAdd/src/vectorAdd.cpp
   :language: C++
   :caption: vectorAdd.cpp

We recommend using CMake for integrating alpaka into your own project.

The following example shows a minimal example of a ``CMakeLists.txt`` that uses alpaka:

Use alpaka via ``add_subdirectory``
-----------------------------------

The ``add_subdirectory`` method does not require alpaka to be installed. Instead, the alpaka project folder must be part of your project hierarchy. The following example expects alpaka to be found in the ``project_path/thirdParty/alpaka``:

.. code-block:: cmake
   :caption: CMakeLists.txt

   cmake_minimum_required(VERSION 3.25)
   project("vectorAdd" CXX)

   add_subdirectory("<path to alpaka>" "${CMAKE_BINARY_DIR}/alpaka")

   add_executable(${PROJECT_NAME} vectorAdd.cpp)
   target_link_libraries(${PROJECT_NAME} PUBLIC alpaka::alpaka)
   alpaka_finalize(${PROJECT_NAME})

In the CMake configuration phase of the project, you must activate the accelerator you want to use:

.. code-block:: bash

    cd <path/to/the/project/root>
    mkdir build && cd build
    cmake .. -Dalpaka_DEP_CUDA=ON
    cmake --build .
    ./vectorAdd

.. A complete list of CMake flags for the  accelerator can be found :doc:`here </advanced/cmake>`.

If the configuration was successful and CMake found the CUDA SDK, the C++ api `cuda` and the executor `gpuCuda` is available.
