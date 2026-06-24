# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from utils import parse_bashi_row
from alpaka_bashi.alpaka_filter import get_valid_compiler_backend_combinations
import bashi


class TestAlpakaFilterValidGetBackendCombination(unittest.TestCase):
    def test_enabled_serial_cpu_is_always_required(self):
        input_row = parse_bashi_row([(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)])
        self.assertEqual(
            len(get_valid_compiler_backend_combinations(input_row)),
            0,
            "Each compiler backend combination requires an enabled serial CPU backend. "
            "Therefore no combination with a disabled serial cpu backend can exist.",
        )

    def test_cpu_compiler_and_serial_cpu(self):
        input_row = parse_bashi_row(
            [(DEVICE_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)]
        )
        self.assertGreater(len(get_valid_compiler_backend_combinations(input_row)), 0)

    def test_nvcc_compiler_and_hip_backend(self):
        input_row = parse_bashi_row(
            [(DEVICE_COMPILER, NVCC, 12.4), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)]
        )
        self.assertEqual(len(get_valid_compiler_backend_combinations(input_row)), 0)

    def test_gcc_host_and_icpx_backend(self):
        input_row = parse_bashi_row([(HOST_COMPILER, GCC, 12), (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON)])
        self.assertEqual(len(get_valid_compiler_backend_combinations(input_row)), 0)

    def test_hipcc_compiler_and_disabled_cpu_backend(self):
        input_row = parse_bashi_row(
            [(DEVICE_COMPILER, HIPCC, 6.3), (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF)]
        )
        self.assertGreater(len(get_valid_compiler_backend_combinations(input_row)), 0)

        input_row_2 = parse_bashi_row(
            [
                (DEVICE_COMPILER, HIPCC, 5.0),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF),
                (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
            ]
        )
        self.assertGreater(len(get_valid_compiler_backend_combinations(input_row_2)), 0)

    def test_icpx_compiler_and_different_sycl_backends_invalid(self):
        for sycl_backend in ONE_API_BACKENDS:
            with self.subTest(sycl_backend=sycl_backend):
                other_sycl_backends = [
                    (other_back, OFF)
                    for other_back in ONE_API_BACKENDS
                    if other_back != sycl_backend
                ]
                input_row = parse_bashi_row(
                    [
                        (DEVICE_COMPILER, ICPX, "2025.0.4"),
                        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF),
                        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
                    ]
                    + other_sycl_backends
                    + [(sycl_backend, ON)]
                )
                self.assertEqual(
                    len(get_valid_compiler_backend_combinations(input_row)),
                    1,
                )
        # icpx can be used for the three different one api backends
        self.assertEqual(
            len(
                get_valid_compiler_backend_combinations(
                    parse_bashi_row(
                        [(HOST_COMPILER, ICPX, "2025.0.4")],
                    )
                )
            ),
            len(ONE_API_BACKENDS),
        )

    def test_cuda_backend_valid_(self):
        input_row = parse_bashi_row(
            [
                (DEVICE_COMPILER, NVCC, 12.4),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4),
            ]
        )
        self.assertEqual(
            len(get_valid_compiler_backend_combinations(input_row)),
            2,  # host compiler GCC and Clang
        )

    GROWING_ROW = [
        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
        (DEVICE_COMPILER, GCC, 8),
        (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
        (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
        (HOST_COMPILER, GCC, 8),
        (ALPAKA_ACC_GPU_CUDA_ENABLE, ON),
    ]

    def test_growing_row(self):
        for i in range(1, len(self.GROWING_ROW) + 1):
            row = parse_bashi_row(self.GROWING_ROW[:i])
            with self.subTest(row=row):
                # The last element of parameter_values invalidates all possible backend combinations.
                if len(row) < len(self.GROWING_ROW):
                    self.assertGreater(
                        len(get_valid_compiler_backend_combinations(row)),
                        0,
                        f"\nrow: {bashi.get_str_row_nice(row)}",
                    )
                else:
                    self.assertEqual(
                        len(get_valid_compiler_backend_combinations(row)),
                        0,
                        f"\nrow: {bashi.get_str_row_nice(row)}",
                    )
