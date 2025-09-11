# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.filter
from alpaka_bashi.globals import CompilerBackendComb
import packaging.version

TEST_CUDA_VER = packaging.version.parse("12.4")


class TestNoHIPBackend(unittest.TestCase):
    def test_no_hip_compiler_backends_valid(self):
        for backend_list in [
            [CompilerBackendComb(GCC, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE])],
            [
                CompilerBackendComb(
                    GCC,
                    GCC,
                    [
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
                    ],
                )
            ],
            [
                CompilerBackendComb(
                    CLANG,
                    CLANG,
                    [
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
                    ],
                ),
                CompilerBackendComb(
                    CLANG_CUDA,
                    CLANG_CUDA,
                    [
                        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
                        ALPAKA_ACC_GPU_CUDA_ENABLE,
                    ],
                ),
            ],
        ]:

            self.assertTrue(
                alpaka_bashi.filter.no_hip_compiler_backends(backend_list),
                backend_list,
            )

    def test_no_hip_compiler_backends_invalid(self):
        for backend_list in [
            [CompilerBackendComb(HIPCC, HIPCC, [ALPAKA_ACC_GPU_HIP_ENABLE])],
            [
                CompilerBackendComb(
                    CLANG,
                    CLANG,
                    [
                        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
                    ],
                ),
                CompilerBackendComb(HIPCC, HIPCC, [ALPAKA_ACC_GPU_HIP_ENABLE]),
            ],
        ]:

            self.assertFalse(
                alpaka_bashi.filter.no_hip_compiler_backends(backend_list),
                backend_list,
            )
