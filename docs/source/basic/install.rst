.. highlight:: bash

Installation
============

**Installing dependencies**

*alpaka* requires a modern C++ compiler (g++, clang++, nvcc, icpx).
**CMake** is the preferred system for configuring the build tree, building and installing.

In order to install **CMake**:

On Linux:

.. code-block::

  # RPM
  sudo dnf install cmake
  # DEB
  sudo apt install cmake

On macOS or Windows:

Download the installer from https://cmake.org/download/

**Dependencies to use specific backends**: Depending on your target platform you may need additional packages to compile and run alpaka.

- NVIDIA GPUs: CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
- AMD GPUs: ROCm / HIP (https://rocmdocs.amd.com/en/latest/index.html)
- Intel GPUs: OneAPI Toolkit (https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)

alpaka as Dependency in Your Application
++++++++++++++++++++++++++++++++++++++++

There are two ways to integrate *alpaka* as dependency with CMake into your project.

.. _install-alpaka:

Install alpaka
~~~~~~~~~~~~~~

The installation must be performed once and can be reused until you require a new version of *alpaka*.
If you only install *alpaka* without enabling tests, examples or benchmarks CMake will not check for compilers, this is postponed to the usage of the installation together with our application.

.. code-block:: bash

  # Clone alpaka from github.com
  git clone --branch dev_v3.x https://github.com/alpaka-group/alpaka.git
  mkdir build && cd build
  # assumed folder structure
  # ├── alpaka
  # └── build
  # if not set, the default is CMAKE_INSTALL_PREFIX=/usr/local
  cmake -DCMAKE_INSTALL_PREFIX=/install/alpaka ..
  cmake --install .

The next CMake code snippet shows you how to integrate the installed *alpaka* library in your application.
You now have the possibility to select the compiler and enable dependencies to target different compute devices.
Examples and brief description can be found in the follow up section :ref:`tests-and-examples`

.. code-block:: cmake

  find_package(alpaka REQUIRED)
  # ...
  add_executable(myTarget src/my.cpp)
  target_link_libraries(myTarget PUBLIC alpaka::alpaka)
  alpaka_finalize(myTarget)
  # ...

The next commands should be executed in the terminal and assume that you prepend the environment variable ``CMAKE_PREFIX_PATH``
with the path `/install/alpaka`, that CMake knows where *alpaka* can be found.

.. code-block:: bash

  cmake <path_to_you_appliction>
  cmake --build . --parallel

Use the Source Code Without Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With CMake it is possible to use *alpaka* without installation, you can simply point to the cloned folder.

.. code-block:: bash

  # Clone alpaka from github.com
  git clone --branch dev_v3.x https://github.com/alpaka-group/alpaka.git

The next CMake code is very similar to the usage of an installed alpaka version with the difference that you should use ``add_subdirectory()`` instead of CMake's ``find_package()``

.. code-block:: cmake

  # `${CMAKE_BINARY_DIR}/alpaka` places the alpaka code in your build directory in the sub directory `alpaka`
  add_subdirectory("<path_to_cloned_alpaka>" "${CMAKE_BINARY_DIR}/alpaka")
  # ...
  add_executable(myTarget src/my.cpp)
  target_link_libraries(myTarget PUBLIC alpaka::alpaka)
  alpaka_finalize(myTarget)
  # ...

You should prepend the environment variable ``CMAKE_PREFIX_PATH`` with the path to the cloned *alpaka* folder.

.. code-block:: bash

  cmake <path_to_you_appliction>
  cmake --build . --parallel

  
Use FetchContent to Download alpaka Automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
CMake's ``FetchContent`` module provides a convenient way to automatically download and integrate *alpaka* into your project at configure time, without requiring manual cloning or installation.

This approach is ideal for:

- Jumpstarting a new alpaka project
- Quick prototyping and testing
- Projects that want to ensure consistent alpaka versions
- CI/CD pipelines where automatic dependency management is preferred

.. warning::
  
    When using FetchContent, *alpaka* will be downloaded each time you run CMake in a clean build directory. For faster builds in development, consider using the installation method or adding the alpaka source subdirectory (for example to your version control system as a submodule/subtree).

.. note::

    Users can directly copy and take over the ``CMakeLists.txt`` and ``main.cpp`` provided below, to start using *alpaka* in their project with ``FetchContent``.   

Complete CMakeLists.txt Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following ``CMakeLists.txt`` demonstrates how to use ``FetchContent`` with *alpaka* and one possible way to configure device specifications:
  
.. literalinclude:: ../../snippets/fetchContent/CMakeLists.txt
    :language: C++
    :caption: CMakeLists.txt
  
Example Source Code
^^^^^^^^^^^^^^^^^^^
Create a ``main.cpp`` file that uses the device specification passed from CMake:
  
.. literalinclude:: ../../snippets/fetchContent/main.cpp
    :language: C++
    :caption: main.cpp

  
Building Your Application
^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
    
    If the ``main.cpp`` and ``CMakeLists.txt`` are placed in an empty directory, then from inside this directory, users can directly follow the code below to build and run the example.

To build and run your application with the default CPU backend:

.. code-block:: bash
  
    mkdir build && cd build
    cmake ..
    cmake --build . --parallel
    ./fetchContentExample
  
Selecting Different Devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can select different device specifications at CMake configuration time using the ``myApp_DEVICE_SPEC`` cache variable, for example 
  
.. code-block:: bash

    # NVIDIA CUDA DEVICE SPEC
    cmake .. -DmyApp_DEVICE_SPEC="cuda:nvidiaGpu"
    # build and run
    cmake --build . --parallel
    ./fetchContentExample
    
.. note::
  
    The device specification system allows you to select the target device at CMake configuration time. The format is ``"api:deviceKind"``, where:
    
    - **api**: The parallel programming interface (``host``, ``cuda``, ``hip``, ``oneApi``)
    - **deviceKind**: The type of device (``cpu``, ``nvidiaGpu``, ``amdGpu``, ``intelGpu``)
    
    Available combinations are: ``host:cpu``, ``cuda:nvidiaGpu``, ``hip:amdGpu``, ``oneApi:cpu``, ``oneApi:intelGpu``, ``oneApi:nvidiaGpu``, ``oneApi:amdGpu``
  
.. warning::
    
    The CUDA, HIP, or Intel backends are only working if the CUDA SDK, HIP SDK, or OneAPI SDK are available respectively


.. _tests-and-examples:

Tests and Examples
++++++++++++++++++

The examples and tests can be compiled before the installation of alpaka see :ref:`install-alpaka`.
They will use alpaka headers from the source directory.
The recipes shown here assume you have installed spack packages for specific compiler versions and that alpaka is available relative to the build folder.

**Load dependencies**

.. code-block::

  spack load gcc@14.1.0
  spack load cmake@3.29.1

**Build and run examples:**

.. code-block::

  # In a directory beside the alpaka source directory (for example in "build-alpaka")
  # ├── alpaka
  # └── build-alpaka
  # ..
  cmake ../alpaka -Dalpaka_EXAMPLES=ON
  cmake --build . --parallel -t vectorAdd
  ./example/vectorAdd/vectorAdd # execution

**Build and run tests:**

.. code-block::

  # ..
  cmake ../alpaka -Dalpaka_TESTS=ON
  cmake --build . --parallel
  ctest

**Enable accelerators:**

alpaka uses different APIs and executors to run kernels on different processors.
To use a specific accelerator in alpaka, two steps are required.

1. Enable the accelerator during the CMake configuration time of the project.
2. Select a specific accelerator in the source code.

By default, only the host API is available which allows to use the executor 'serial' to run on CPUs without using multithreading.
If OpenMP is available on the system, additionally the executor `cpuOmpBlocks` can be used to run on all cores of the CPU.
If oneAPI TBB is available on the system, additionally the executor `cpuTbbBlocks` can be used to run on all cores of the CPU.
CMake option `alpaka_DEP_*` controls whether a parallelization framework is used and introduces a dependency on third-party libraries.
This allows the usage of the corresponding executor e.g. `gpuCuda`, `gpuHip` or `oneApi`
`alpaka_EXEC_*` activates or deactivates which execution schemes will be used for examples/tests and benchmarks.

**compile for CPU only (serial and OpenMP):**

.. code-block::

  spack load gcc@14.1.0
  spack load cmake@3.29.1

  # -Dalpaka_DEP_OMP=ON is implicitly set, if the compiler not support OpenMP only serial code will be generated
  # Assuming alpaka source is in ../alpaka with respect to the current directory
  #
  cmake ../alpaka -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON
  cmake --build . --parallel
  ctest --output-on-failure

**compile for CPU with (serial and oneAPI TBB):**

.. code-block::

  spack load cmake@3.29.1
  spack load intel-oneapi-tbb@2021.10.0

  cmake ../alpaka -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_TBB=ON -Dalpaka_DEP_OMP=OFF
  cmake --build . --parallel
  ctest --output-on-failure

**compile for NVIDIA CUDA only:**

.. code-block::

  spack load cmake@3.29.1
  spack load cuda@12.4.0

  # use -DCMAKE_CUDA_ARCHITECTURES=80 to set the GPU architecture
  cmake ../alpaka -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_DEP_CUDA=ON -Dalpaka_EXEC_CpuSerial=OFF
  cmake --build . --parallel
  ctest --output-on-failure

**compile for AMD HIP only:**

.. code-block::

  spack load cmake@3.29.1
  spack load hip@6.3.4
  export CXX=clang++

  # use -DCMAKE_HIP_ARCHITECTURES=gfx906 to set the GPU architecture
  # for older CMake version sometimes the architecture must be set with -DAMDGPU_TARGETS=gfx906
  cmake ../alpaka -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_DEP_HIP=ON -Dalpaka_EXEC_CpuSerial=OFF
  cmake --build . --parallel
  ctest --output-on-failure

**compile for OneApi SYCL CPU/GPU only**

.. code-block::

  spack load cmake@3.29.1
  spack load intel-oneapi-compilers@2025.0.4
  spack load intel-oneapi-dpl@2022.2.0

  # by default all device kinds get be addressed CPU and Intel GPU will be addressed
  # You can add support for NVIDIA GPU and AMD GPU via
  #  -Dalpaka_ONEAPI_AmdGpu=ON
  #  -Dalpaka_ONEAPI_NvidiaGpu=ON
  # or deselect the device kind with
  #  -Dalpaka_ONEAPI_Cpu=OFF
  #  -Dalpaka_ONEAPI_IntelGpu=OFF
  #
  # The architecture can be set with -Dalpaka_ONEAPI_*_ARCH=<arch>
  # Cpu ISA e.g. avx,avx2, avx512
  # Nvidia only the sm number is needed e.g. 80
  # Amd full qualifier is required e.g. gfx906
  cmake ../alpaka -DCMAKE_CXX_COMPILER=icpx -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_EXAMPLES=ON -Dalpaka_DEP_ONEAPI=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_EXEC_CpuSerial=OFF
  cmake --build . --parallel
  ctest --output-on-failure

.. warning::

  If an api is selected in the source code that is not activated during CMake configuration time, a compiler error occurs.
