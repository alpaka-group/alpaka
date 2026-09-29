# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import io
import unittest

from bashi.globals import ALPAKA_ACC_GPU_CUDA_ENABLE, CLANG, CLANG_CUDA, CMAKE, DEVICE_COMPILER, HOST_COMPILER
from utils import parse_bashi_row

from alpaka_bashi.alpaka_filter import AlpakaFilter, _pretty_name_compiler, check_clang_cuda_cmake_support_a4


class TestClangHostCompilerCUDAsdk(unittest.TestCase):
    VALID_ROWS = [
        [(HOST_COMPILER, CLANG, 17), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.3")],
        [(HOST_COMPILER, CLANG_CUDA, 22), (CMAKE, "3.30")],
        [(DEVICE_COMPILER, CLANG_CUDA, 20), (CMAKE, "3.27")],
        [(DEVICE_COMPILER, CLANG_CUDA, 20), (CMAKE, "3.31")],
        [(DEVICE_COMPILER, CLANG_CUDA, 20), (CMAKE, "4.5")],
        [(DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "3.31")],
        [(DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "4.5")],
        [(DEVICE_COMPILER, CLANG_CUDA, 27), (CMAKE, "3.31")],
        [(HOST_COMPILER, CLANG_CUDA, 27), (CMAKE, "4.2")],
    ]

    def test_valid_check_clang_cuda_cmake_support_a4(self):
        for row in self.VALID_ROWS:
            with self.subTest(row=row):
                self.assertTrue(
                    check_clang_cuda_cmake_support_a4(parse_bashi_row(row), AlpakaFilter()),
                    f"{row}",
                )
                self.assertTrue(AlpakaFilter()(parse_bashi_row(row)), f"{row}")

    INVALID_ROWS = [
        [(DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "3.27")],
        [(CMAKE, "3.30.1"), (HOST_COMPILER, CLANG_CUDA, 23)],
        [(CMAKE, "3.30.2"), (HOST_COMPILER, CLANG_CUDA, 27)],
    ]

    def test_invalid_check_clang_cuda_cmake_support_a4(self):
        for row in self.INVALID_ROWS:
            with self.subTest(row=row):
                reason_msg_func = io.StringIO()

                parsed_row = parse_bashi_row(row)
                self.assertFalse(
                    check_clang_cuda_cmake_support_a4(parsed_row, AlpakaFilter(output=reason_msg_func)),
                    f"{row}",
                )

                if DEVICE_COMPILER in parsed_row:
                    compiler_type = DEVICE_COMPILER
                else:
                    compiler_type = HOST_COMPILER

                expected_error_msg = (
                    f"CMAKE {parsed_row[CMAKE].version} does not support "
                    f"{_pretty_name_compiler(compiler_type)} Clang-Cuda {parsed_row[compiler_type].version}"
                )

                self.assertEqual(reason_msg_func.getvalue(), expected_error_msg, f"{row}")

                reason_msg_filter = io.StringIO()
                self.assertFalse(AlpakaFilter(output=reason_msg_filter)(parsed_row), f"{row}")
                self.assertEqual(reason_msg_filter.getvalue(), expected_error_msg, f"{row}")
