.. highlight:: bash

alpaka Installation
===================

* Clone alpaka from github.com

.. code-block::

  git clone https://github.com/alpaka-group/alpaka
  cd alpaka

* Install alpaka without tests and examples

.. code-block::

  # git clone https://github.com/alpaka-group/alpaka
  # cd alpaka
  mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/install/ \
        -Dalpaka_BUILD_EXAMPLES=OFF      \
        -DBUILD_TESTING=OFF              \
        ..
  make install

* Configure Accelerators

.. code-block::

  # ..
  cmake -DALPAKA_ACC_GPU_CUDA_ENABLE=ON ..

* Just build an example

.. code-block::

  # ..
  cmake -Dalpaka_BUILD_EXAMPLES=ON ..
  make vectorAdd
  ./example/vectorAdd/vectorAdd # execution
