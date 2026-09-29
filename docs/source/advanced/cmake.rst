CMake
=====

alpaka configures a large part of its functionality at compile time. Therefore, a lot of compiler and link flags are needed, which are set by CMake arguments. First, we show a simple way to build alpaka for different back-ends using `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_. The second part of the documentation shows the general and back-end specific alpaka CMake flags.

.. hint::

   To display the cmake variables with value and type in the build folder of your project, use ``cmake -LH <path-to-build>``.

Presets
-------

The `CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_ are defined in the ``CMakePresets.json`` file. Each preset contains a set of CMake arguments. We use different presets to build the examples and tests with different back-ends. Execute the following command to display all presets:

.. code-block:: bash

   cd <alpaka_project_root>
   cmake --list-presets

To configure, build and run the tests of a specific preset, run the following commands (for the example, we use the ``rel-host-cpu-gcc`` preset):

.. code-block:: bash

   cd <alpaka_project_root>
   # configure a specific preset
   cmake --preset rel-host-cpu-gcc
   # build the preset
   cmake --build --preset rel-host-cpu-gcc
   # run test of the preset
   ctest --preset rel-host-cpu-gcc

All presets are configured and built in a subfolder of the ``<alpaka_project_root>/build`` folder.

Modifying and Extending Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to change a preset is to set CMake arguments during configuration:

.. code-block:: bash

   cd <alpaka_project_root>
   # configure the rel-host-cpu-gcc preset with clang++ as C++ compiler
   cmake --preset rel-host-cpu-gcc -DCMAKE_CXX_COMPILER=clang++
   # build the preset
   cmake --build --preset rel-host-cpu-gcc
   # run test of the preset
   ctest --preset rel-host-cpu-gcc

It is also possible to configure the default setting first and then change the arguments with ``ccmake``:

.. code-block:: bash

   cd <alpaka_project_root>
   # configure the rel-host-cpu-gcc preset with clang++ as C++ compiler
   cmake --preset rel-host-cpu-gcc
   cd build/rel-host-cpu-gcc
   ccmake .
   cd ../..
   # build the preset
   cmake --build --preset rel-host-cpu-gcc
   # run test of the preset
   ctest --preset rel-host-cpu-gcc

CMake presets also offer the option of creating personal, user-specific configurations based on the predefined CMake presets. To do this, you can create the file ``CMakeUserPresets.json`` in the root directory of your project (the file is located directly next to ``CMakePresets.json``). You can then create your own configurations from the existing CMake presets. The following example takes the rel-host-cpu-gcc configuration, uses ``ninja`` as the generator instead of the standard generator and uses the build type ``RELEASE``.

.. code-block:: json

   {
      "version": 3,
      "cmakeMinimumRequired": {
        "major": 3,
        "minor": 25,
        "patch": 0
      },
      "configurePresets": [
        {
            "name": "rel-host-cpu-gcc-ninja-release",
            "inherits": "rel-host-cpu-gcc",
            "generator": "Ninja",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": {
                    "type": "STRING",
                    "value": "RELEASE"
                }
            }
        }
      ]
   }

.. hint::

   Many IDEs like `Visual Studio Code <https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/cmake-presets.md>`_ and `CLion <https://www.jetbrains.com/help/clion/cmake-presets.html>`_ support CMake presets.

Arguments
---------

.. tip::

    The executor ``exec::cpuSerial`` is always available and does not require any special CMake flags beside linking the target ``alpaka::alpaka`` or ``alpaka::host``.


``alpaka_CXX_STANDARD``
  .. code-block:: markdown

     Set the C++ standard version.


``alpaka_TESTS``
  .. code-block:: markdown

     Build the testing tree with the current environment.
     This option is available during the installation of alpaka.
     Additionally this option can be activated at the moment where an already installed alpaka version is used
     within your project, since the environment can be different to the one used during the installation of alpaka.

``alpaka_LOG_*``

  Enable and configure logging, see :ref:`dev-logging`.

``alpaka_SYSTEM_CATCH2``
  .. code-block:: markdown

     If `ON` use Catch2 to pre-installed on system else fetch Catch2 via CMake's fetch content.
     This option is only available of `alpaka_TESTS` or `alpaka_BENCHMARKS` is `ON`.

``alpaka_FAST_MATH``
  .. code-block:: markdown

     Enable fast-math in kernels.

``alpaka_FTZ``
  .. code-block:: markdown

     Set flush to zero for GPU.

``alpaka_COMPILE_PEDANTIC``
  .. code-block:: markdown

     If `ON` strict compile flags will be used which we apply typically only in the CI.
     Enables `-Wall`, '-Werror`, ...

``alpaka_RELOCATABLE_DEVICE_CODE``
  .. code-block:: markdown

     Enable relocatable device code.
     This options affects only HIP, CUDA and oneAPI.
     Note: This affects all targets in the
     CMake scope where `alpaka_RELOCATABLE_DEVICE_CODE` is set. For the
     effects on CUDA code see NVIDIA's blog post:

  https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

``alpaka_SIMD``
  .. code-block:: markdown

     Valid values `DEFAULT`,`STDSIMD` or `EMULATION`.
     - `DEFAULT`   - checks for `std::simd` support of the CXX compiler, if supported it used `std::simd` for all supported value types, else falls back to emulated SIMD packs.
     - `STDSIMD`   - enforce that the CXX compiler supports `std::simd`, if not CMake will fail
     - `EMULATION` - disable the usage of `std::simd` even if supported by the CXX compiler

``alpaka_LANGUAGE_SEPARATION``
  .. code-block:: markdown

     Enable that C++ files for CUDA and HIP will be copied to a unique folder and compiled from there, instead of using the original file.
     If this option is set to ``OFF`` and the same C++ file is used twice in the project within two different CMake targets where one is linking against ``alpaka::host`` and the other to ``alpaka::cuda`` or ``alpaka::hip`` both will use the CUDA/HIP compiler for compilation.
     In the default case where the option is set to ``ON`` one target is compiled with the native CXX compiler and the other with the CUDA/HIP compiler.
     CUDA/HIP compiler have sometimes problems to compile any C++ code, therefore try to avoid setting this option to ``OFF`` because you lose the possibility to root heterogeneous code to the native CXX compiler.

     **Attention:** Only when this option is enabled can the same source code file be compiled and linked for multiple APIs (CUDA, HIP, ...) within the same build.

Host
^^^^

``alpaka_HOST_Cpu``
  .. code-block:: markdown

     Enable/Disable the `api::host` device `deviceKind::cpu`.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_Host_Cpu` to disable host CPU device support.

``alpaka_HOST_NumaCpu``
  .. code-block:: markdown

     Enable/Disable the `api::host` device `deviceKind::numaCpu`.
     Requires the dependency `hwloc`, can be enabled with the CMake option `-Dalpaka_DEP_HWLOC=ON`.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_Host_NumaCpu` to disable host NUMA CPU device support.

CUDA
^^^^

To enable the CUDA back-end please extend ``CMAKE_PREFIX_PATH`` with the path to the CUDA installation.

``alpaka_DEP_CUDA``
  .. code-block:: markdown

     Enable the CUDA API and the usage of the executor `exec::gpuCuda`.

``alpaka_CUDA_NvidiaGpu``
  .. code-block:: markdown

     Enable/Disable the `api::cuda` device `deviceKind::nvidiaGpu`.
     Requires the dependency `hwloc`.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_Cuda_NvidiaGpu` to disable CUDA NVIDIA GPU device support.

``CMAKE_CUDA_ARCHITECTURES``
  .. code-block:: markdown

     Set the GPU architecture: e.g. "80".
     Supports a list of architectures separated by semicolons, e.g. "80;86;89;90".

``CMAKE_CUDA_COMPILER``
  .. code-block:: markdown

     Set the CUDA compiler: "nvcc" (default) or "clang++".

``CMAKE_CUDA_HOST_COMPILER``
  .. code-block:: markdown

     Select a specific CUDA host compiler, it can differ from the `CMAKE_CXX_COMPILER`.

``alpaka_CUDA_KEEP_FILES``
  .. code-block:: markdown

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.

``alpaka_CUDA_EXPT_EXTENDED_LAMBDA``
  .. code-block:: markdown

     Enable experimental, extended host-device lambdas in NVCC.

``alpaka_CUDA_SHOW_CODELINES``
  .. code-block:: markdown

     Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES
     is enabled source code will be inlined in ptx.
     One of the added flags is: --generate-line-info

``alpaka_CUDA_SHOW_REGISTER``
  .. code-block:: markdown

     Show the number of used kernel registers during compilation and create PTX.

HIP
^^^

``alpaka_DEP_HIP``
  .. code-block:: markdown

     Enable the HIP API and the usage of the executor `exec::gpuHip`.

``alpaka_HIP_AmdGpu``
  .. code-block:: markdown

     Enable/Disable the `api::hip` device `deviceKind::amdGpu`.
     Requires the dependency `hwloc`.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_Hip_AmdGpu` to disable HIP AMD GPU device support.

``CMAKE_HIP_ARCHITECTURES``
  .. code-block:: markdown

     Set the GPU architecture: e.g. "gfx90a".
     Supports a list of architectures separated by semicolons, e.g. "gfx900;gfx906;gfx908".

A list of the GPU architectures can be found `here <https://llvm.org/docs/AMDGPUUsage.html#processors>`_.

``alpaka_HIP_KEEP_FILES``
  .. code-block:: markdown

     Keep all intermediate files that are generated during internal compilation
     steps 'CMakeFiles/<targetname>.dir'.

oneAPI SYCL
^^^^^^^^^^^

``alpaka_DEP_ONEAPI``
  .. code-block:: markdown

     Enable SYCL oneAPI support and the usage of the executor `exec::oneApi`.

``alpaka_ONEAPI_Cpu``
  .. code-block:: markdown

     Enable SYCL oneAPI CPU device support.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_OneApi_Cpu` to disable SYCL oneAPI CPU device support.

``alpaka_ONEAPI_IntelGpu``
  .. code-block:: markdown

     Enable SYCL oneAPI Intel GPU device support.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_OneApi_IntelGpu` to disable SYCL oneAPI Intel GPU device support.

``alpaka_ONEAPI_NvidiaGpu``
  .. code-block:: markdown

     Enable SYCL oneAPI Nvidia GPU device support.
     Requires the oneAPI compiler extension for CUDA support.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_OneApi_NvidiaGpu` to disable SYCL oneAPI NVIDIA GPU device support.

``alpaka_ONEAPI_AmdGpu``
  .. code-block:: markdown

     Enable SYCL oneAPI AMD GPU device support.
     Requires the oneAPI compiler extension for AMD ROCM support.
     This option effects the entries of the list `onHost::enabledDeviceSpecs` typically used in alpaka examples/benchmarks and tests.

     **without CMake:** Define the preprocessor define `ALPAKA_DISABLE_OneApi_AmdGpu` to disable SYCL oneAPI AMD GPU device support.

OpenMP
^^^^^^

``alpaka_DEP_OMP``
  .. code-block:: markdown

     Enable OpenMP CPU acceleration and the usage of the executor `exec::cpuOmpBlocks`.
     Note: OpenMP offloading is currently not supported.

Numa Awareness
^^^^^^^^^^^^^^

``alpaka_DEP_HWLOC``
  .. code-block:: markdown

     Enable/Disable the possibility to use the `host` device kind `numaCpu`.
     `AUTO` is the default and will check if `hwloc` is installed and if found handle `alpaka_DEP_HWLOC` as if it is set to `ON`, else `OFF`.
     If you installed `hwloc` by hand you should extent the environment variable `PKG_CONFIG_PATH` to the directory of `hwloc.pc` that CMake can find the dependency.

     This device is seeing each NUMA domain with all directly connected CPU cores as a single device.
     This can avoid false data sharing and can speedup your code.
     You should utilize all NUMA devices to get the full performance of your CPU, this requires manual a domain decomposition of your problem to be able to run it on all NUMA devices.
     If you use a blocking host queue together with the serial executor the executing thread will only be pinned if it is not the thread of the process.

     **without CMake:** If the header `hwloc.h` is in the include path you should link `-lhwloc` or disable `hwloc` support by defining the preprocessor define `ALPAKA_DISABLE_HWLOC`.

``alpaka_HOST_MemPinningCanFail``
  .. code-block:: markdown

     Enable/Disable that the memory pinning via `hwloc` can fail.
     On a system e.g. in container environments where it is not allowed to pin memory, this option can be enabled to allow that pinning can fail without an exception.
     Requires the dependency `hwloc`.

     **without CMake:**  Defining the preprocessor define `ALPAKA_HOST_MEM_PINNING_CAN_FAIL` to allow that memory pinning can fail without an exception.

Intel oneAPI Threading Building Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``alpaka_DEP_TBB``
  .. code-block:: markdown

     Enable TBB CPU acceleration and the usage of the executor `exec::cpuTbbBlocks`.

Executors
^^^^^^^^^

  With the following flags you can choose the available executors within the compile time C++ list ``exec::enabledExecutors``.
  ``exec::allExecutors`` will always have all alpaka executors listed independent of the chosen CMake flags.

``alpaka_EXEC_CpuSerial``
  .. code-block:: markdown

     Enable the serial CPU executor `exec::cpuSerial`.

``alpaka_EXEC_CpuOmpBlocks``
  .. code-block:: markdown

     Enable the OpenMP CPU blocks executor `exec::cpuOmpBlocks`.

``alpaka_EXEC_TbbBlocks``
  .. code-block:: markdown

     Enable the TBB CPU blocks executor `exec::cpuTbbBlocks`.

``alpaka_EXEC_GpuCuda``
  .. code-block:: markdown

     Enable the CUDA GPU executor `exec::gpuCuda`.

``alpaka_EXEC_GpuHip``
  .. code-block:: markdown

     Enable the HIP GPU executor `exec::gpuHip`.

``alpaka_EXEC_OneApi``
  .. code-block:: markdown

     Enable the oneAPI SYCL executor `exec::oneApi`.


Available during the Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following options are only available during the installation of alpaka, i.e., when building the alpaka library itself.

``alpaka_EXAMPLES``
  .. code-block:: markdown

     Build the examples.

``alpaka_BENCHMARKS``
  .. code-block:: markdown

     Build the benchmarks.

``alpaka_HEADERCHECKS``
  .. code-block:: markdown

     Validate headers to ensure that all necessary includes are present.

``alpaka_DOCS``
  .. code-block:: markdown

     Build documentation code snippets.

Code Sanitizers for CPU
^^^^^^^^^^^^^^^^^^^^^^^

``alpaka_ASAN``
  .. code-block:: markdown

     Enable address sanitizer.

``alpaka_TSAN``
  .. code-block:: markdown

     Enable thread sanitizer.

``alpaka_UBSAN``
  .. code-block:: markdown

      Enable undefined behavior sanitizer.

``alpaka_LSAN``
  .. code-block:: markdown

      Enable memory leak sanitizer.

Targets
-------

alpaka is providing CMake targets based on the optional activated dependencies ``alpaka_DEP_*``

.. _alpaka-headers:

``alpaka::headers``
^^^^^^^^^^^^^^^^^^^

- Set include dependencies.
- Activate header code based on the CMake option ``alpaka_EXEC_*`` (required for examples/test/benchmarks).
- Set the ``CXX`` standard.

.. _alpaka-host:

``alpaka::host``
^^^^^^^^^^^^^^^^

- Links :ref:`alpaka-headers`.
- Sets compiler flags that are generic for CUDA/HIP and C++ targets.
- Sets special compiler flags that are for C++ targets only.

.. _alpaka-cuda:

``alpaka::cuda``
^^^^^^^^^^^^^^^^

- Available if the CMake option ``-Dalpaka_DEP_CUDA=ON`` is set.
- Activates support for NVIDIA GPUs.

.. _alpaka-hip:

``alpaka::hip``
^^^^^^^^^^^^^^^

- Available if the CMake option ``-Dalpaka_DEP_HIP=ON`` is set.
- Activates support for AMD GPUs.

.. _alpaka-oneapi:

``alpaka::oneapi``
^^^^^^^^^^^^^^^^^^

- Available if the CMake option ``-Dalpaka_DEP_ONEAPI=ON`` is set.
- Activates support for CPUs, NVIDIA GPUs, AMD GPUs, and Intel GPUs.

.. _alpaka-alpaka:

``alpaka``, ``alpaka::alpaka``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Links :ref:`alpaka-headers` and provides access to APIs activated by dependency switches.
- If at least two of the dependencies ``alpaka_DEP_CUDA``, ``alpaka_DEP_HIP``, or ``alpaka_DEP_ONEAPI`` are activated:

  - **None** of the APIs (except ``host``) will be added because these dependencies are mutually exclusive.
  - You should add the required executor targets manually.

.. note::

   Dependencies of ``alpaka_DEP_OMP`` and ``alpaka_DEP_TBB`` are linked into the target ``alpaka::host`` because they influence only host executors and do not provide additional alpaka API support.
   When enabling `alpaka_DEP_TBB`, make sure Intel oneTBB version 2021.10 or newer (including 2022.x releases) is available.

After linking alpaka targets to your application, you should call ``alpaka_finalize(targetName)`` for each target that uses alpaka.
``alpaka_finalize()`` is a CMake function that ensures all necessary compile definitions and compiler options are set for the given target.
Depending on the enabled dependencies (``alpaka_DEP_*``), this call may copy your source files to a temporary directory and compile them with the appropriate compiler.
Linking non-alpaka targets after calling ``alpaka_finalize()`` is allowed.
You should not include header files using relative paths in your source files.
These relative paths may become invalid after ``alpaka_finalize()`` is called, because the source files can be copied to a different location.

Examples
--------

- standard application enabling APIs depending on the selected cmake dependencies

   .. code-block:: cmake

       # call: cmake -Dalpaka_DEP_CUDA=ON pathToAlpaka
       add_executable(fooTarget src/main.cpp)
       # provides access to host and CUDA API
       target_link_libraries(fooTarget PUBLIC alpaka)
       alpaka_finalize(fooTarget)

- build a shared library

   .. code-block:: cmake

       # call: cmake -Dalpaka_DEP_CUDA=ON pathToAlpaka
       add_library(fooShared SHARED src/foo.cpp)
       target_link_libraries(fooShared PUBLIC alpaka)
       alpaka_finalize(fooShared)

       add_executable(fooTarget src/main.cpp)
       target_link_libraries(fooTarget PRIVATE fooShared)

- standard application which prefers manual selection of the APIs

   .. code-block:: cmake

       # call: cmake -Dalpaka_DEP_CUDA=ON pathToAlpaka
       add_executable(fooTarget src/main.cpp)
       # provides access to host and CUDA API
       target_link_libraries(fooTarget PUBLIC alpaka::headers)
       target_link_libraries(fooTarget PUBLIC alpaka::cuda)
       alpaka_finalize(fooTarget)

- using more than one dependency

   .. code-block:: cmake

       # call: cmake -Dalpaka_DEP_CUDA=ON -Dalpaka_DEP_HIP=ON pathToAlpaka
       add_executable(fooTarget src/main.cpp)
       # provides access to host and cuda API
       # the target alpaka::alpaka is now equal to alpaka::headers
       target_link_libraries(fooTarget PUBLIC alpaka alpaka::cuda)
       alpaka_finalize(fooTarget)

       add_executable(barTarget src/main.cpp)
       # provides access to host and hip API
       target_link_libraries(barTarget PUBLIC alpaka alpaka::hip)
       alpaka_finalize(barTarget)
