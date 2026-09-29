# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import io
import unittest

import packaging.version
from bashi import VersionRelation
from bashi.globals import ALPAKA_ACC_GPU_CUDA_ENABLE, CLANG, CMAKE, DEVICE_COMPILER, GCC, HOST_COMPILER, NVCC, OFF
from bashi.version.dependencies.nvcc import NvccHostSupport
from utils import parse_bashi_row

from alpaka_bashi.alpaka_filter import (
    AlpakaFilter,
    check_clang_host_compiler_supported_cuda_sdk_a2,
    check_clang_host_compiler_supported_nvcc_a3,
    check_if_nvcc_supports_the_host_compiler_a5,
)


class TestClangHostCompilerCUDAsdk(unittest.TestCase):
    VALID_ROWS = [
        [(HOST_COMPILER, CLANG, 17), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.3")],
        [(HOST_COMPILER, CLANG, 9), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.8")],
        [(HOST_COMPILER, CLANG, 12), (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)],
        [(HOST_COMPILER, GCC, 13), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.3")],
    ]

    def test_valid_clang_host_compiler_supported_cuda_sdk_a2(self):
        for row in self.VALID_ROWS:
            with self.subTest(row=row):
                self.assertTrue(
                    check_clang_host_compiler_supported_cuda_sdk_a2(parse_bashi_row(row), AlpakaFilter()),
                    f"{row}",
                )
                self.assertTrue(AlpakaFilter()(parse_bashi_row(row)), f"{row}")

    INVALID_ROWS = [
        [(HOST_COMPILER, CLANG, 17), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.2")],
        [(HOST_COMPILER, CLANG, 9), (ALPAKA_ACC_GPU_CUDA_ENABLE, "12.8")],
        [(HOST_COMPILER, CLANG, 30), (ALPAKA_ACC_GPU_CUDA_ENABLE, "11.0")],
    ]

    def test_invalid_clang_host_compiler_supported_cuda_sdk_a2(self):
        EXPECTED_ERROR_MSG = "Clang as nvcc host compiler is only working since CUDA 13.3."

        for row in self.INVALID_ROWS:
            with self.subTest(row=row):
                reason_msg_func = io.StringIO()
                self.assertFalse(
                    check_clang_host_compiler_supported_cuda_sdk_a2(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg_func)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg_func.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg_filter = io.StringIO()
                self.assertFalse(AlpakaFilter(output=reason_msg_filter)(parse_bashi_row(row)), f"{row}")
                self.assertEqual(reason_msg_filter.getvalue(), EXPECTED_ERROR_MSG, f"{row}")


class TestClangHostCompilerNvcc(unittest.TestCase):
    VALID_ROWS = [
        [(HOST_COMPILER, CLANG, 17), (DEVICE_COMPILER, NVCC, "13.3")],
        [(HOST_COMPILER, CLANG, 9), (DEVICE_COMPILER, NVCC, "13.8")],
        [(HOST_COMPILER, GCC, 13), (DEVICE_COMPILER, NVCC, "13.3")],
    ]

    def test_valid_check_clang_host_compiler_supported_nvcc_a3(self):
        for row in self.VALID_ROWS:
            with self.subTest(row=row):
                self.assertTrue(
                    check_clang_host_compiler_supported_nvcc_a3(parse_bashi_row(row), AlpakaFilter()),
                    f"{row}",
                )
                self.assertTrue(AlpakaFilter()(parse_bashi_row(row)), f"{row}")

    INVALID_ROWS = [
        [(HOST_COMPILER, CLANG, 17), (DEVICE_COMPILER, NVCC, "13.2")],
        [(HOST_COMPILER, CLANG, 9), (DEVICE_COMPILER, NVCC, "12.8")],
        [(HOST_COMPILER, CLANG, 30), (DEVICE_COMPILER, NVCC, "11.0")],
    ]

    def test_invalid_check_clang_host_compiler_supported_nvcc_a3(self):
        EXPECTED_ERROR_MSG = "The Clang host compiler is only working since nvcc 13.3."

        for row in self.INVALID_ROWS:
            with self.subTest(row=row):
                reason_msg_func = io.StringIO()
                self.assertFalse(
                    check_clang_host_compiler_supported_nvcc_a3(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg_func)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg_func.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg_filter = io.StringIO()
                self.assertFalse(AlpakaFilter(output=reason_msg_filter)(parse_bashi_row(row)), f"{row}")
                self.assertEqual(reason_msg_filter.getvalue(), EXPECTED_ERROR_MSG, f"{row}")


class TestNvccMaxSupportedGccHostCompiler(unittest.TestCase):
    NVCC_GCC_SUPPORT: list[NvccHostSupport] = [
        NvccHostSupport("13.4", "16"),
        NvccHostSupport("13.0", "15"),
        NvccHostSupport("12.9", "14"),
        NvccHostSupport("12.6", "13"),
    ]

    VALID_ROWS = [
        [(HOST_COMPILER, GCC, 16), (DEVICE_COMPILER, NVCC, "13.7")],
        [(HOST_COMPILER, GCC, 16), (DEVICE_COMPILER, NVCC, "13.4")],
        [(HOST_COMPILER, GCC, 15), (DEVICE_COMPILER, NVCC, "13.3")],
        [(HOST_COMPILER, GCC, 9), (DEVICE_COMPILER, NVCC, "13.8")],
        [(CMAKE, 3.31), (DEVICE_COMPILER, NVCC, "13.3")],
    ]

    def test_valid_nvcc_gcc_host_compiler_support_a5(self):
        for row in self.VALID_ROWS:
            with self.subTest(row=row):
                alpaka_filter = AlpakaFilter(
                    version_relation=VersionRelation(nvcc_gcc_max_version=self.NVCC_GCC_SUPPORT)
                )
                self.assertTrue(
                    check_if_nvcc_supports_the_host_compiler_a5(parse_bashi_row(row), alpaka_filter),
                    f"{row}",
                )
                self.assertTrue(alpaka_filter(parse_bashi_row(row)), f"{row}")

    INVALID_ROWS = [
        [(HOST_COMPILER, GCC, 17), (DEVICE_COMPILER, NVCC, "13.4")],
        [(HOST_COMPILER, GCC, 20), (DEVICE_COMPILER, NVCC, "13.4")],
        [(HOST_COMPILER, GCC, 17), (DEVICE_COMPILER, NVCC, "12.4")],
    ]

    def test_invalid_nvcc_gcc_host_compiler_support_a5(self):
        for row in self.INVALID_ROWS:
            with self.subTest(row=row):
                parsed_row = parse_bashi_row(row)
                expected_error_msg = (
                    f"There is nvcc which supports gcc {parsed_row[HOST_COMPILER].version} as host compiler"
                )

                reason_msg_func = io.StringIO()
                alpaka_filter = AlpakaFilter(
                    version_relation=VersionRelation(nvcc_gcc_max_version=self.NVCC_GCC_SUPPORT),
                    output=reason_msg_func,
                )
                self.assertFalse(
                    check_if_nvcc_supports_the_host_compiler_a5(parsed_row, alpaka_filter),
                    f"{row}",
                )
                self.assertEqual(reason_msg_func.getvalue(), expected_error_msg, f"{row}")

                reason_msg_filter = io.StringIO()
                alpaka_filter2 = AlpakaFilter(
                    version_relation=VersionRelation(nvcc_gcc_max_version=self.NVCC_GCC_SUPPORT),
                    output=reason_msg_filter,
                )

                self.assertFalse(alpaka_filter2(parsed_row), f"{row}")
                self.assertEqual(reason_msg_filter.getvalue(), expected_error_msg, f"{row}")


class TestNvccMaxSupportedClangHostCompiler(unittest.TestCase):
    NVCC_CLANG_SUPPORT: list[NvccHostSupport] = [
        NvccHostSupport("13.4", "22"),
        NvccHostSupport("13.0", "20"),
        NvccHostSupport("12.9", "19"),
        NvccHostSupport("12.6", "18"),
        NvccHostSupport("12.4", "17"),
    ]

    VALID_ROWS = [
        [(HOST_COMPILER, CLANG, 22), (DEVICE_COMPILER, NVCC, "13.7")],
        [(HOST_COMPILER, CLANG, 22), (DEVICE_COMPILER, NVCC, "13.4")],
        [(HOST_COMPILER, CLANG, 17), (DEVICE_COMPILER, NVCC, "13.3")],
        [(HOST_COMPILER, CLANG, 9), (DEVICE_COMPILER, NVCC, "13.8")],
        [(CMAKE, 3.31), (DEVICE_COMPILER, NVCC, "13.3")],
    ]

    def test_valid_nvcc_clang_host_compiler_support_a5(self):
        for row in self.VALID_ROWS:
            with self.subTest(row=row):
                alpaka_filter = AlpakaFilter(
                    version_relation=VersionRelation(nvcc_clang_max_version=self.NVCC_CLANG_SUPPORT)
                )
                self.assertTrue(
                    check_if_nvcc_supports_the_host_compiler_a5(parse_bashi_row(row), alpaka_filter),
                    f"{row}",
                )
                self.assertTrue(alpaka_filter(parse_bashi_row(row)), f"{row}")

    INVALID_ROWS = [
        [(HOST_COMPILER, CLANG, 23), (DEVICE_COMPILER, NVCC, "13.4")],
        [(HOST_COMPILER, CLANG, 27), (DEVICE_COMPILER, NVCC, "13.4")],
        [(HOST_COMPILER, CLANG, 23), (DEVICE_COMPILER, NVCC, "12.4")],
        [(HOST_COMPILER, CLANG, 24), (DEVICE_COMPILER, NVCC, "12.4")],
    ]

    def test_invalid_nvcc_clang_host_compiler_support_a5(self):

        for row in self.INVALID_ROWS:
            with self.subTest(row=row):
                parsed_row = parse_bashi_row(row)
                expected_error_msg = (
                    f"There is nvcc which supports clang {parsed_row[HOST_COMPILER].version} as host compiler"
                )

                reason_msg_func = io.StringIO()
                alpaka_filter = AlpakaFilter(
                    version_relation=VersionRelation(nvcc_clang_max_version=self.NVCC_CLANG_SUPPORT),
                    output=reason_msg_func,
                )
                self.assertFalse(
                    check_if_nvcc_supports_the_host_compiler_a5(parsed_row, alpaka_filter),
                    f"{row}",
                )
                self.assertEqual(reason_msg_func.getvalue(), expected_error_msg, f"{row}")

                # nvcc version is older than 13.3, rule a3 is triggered first
                if parsed_row[DEVICE_COMPILER].version < packaging.version.parse("13.3"):
                    expected_error_msg = "The Clang host compiler is only working since nvcc 13.3."

                reason_msg_filter = io.StringIO()
                alpaka_filter2 = AlpakaFilter(
                    version_relation=VersionRelation(nvcc_clang_max_version=self.NVCC_CLANG_SUPPORT),
                    output=reason_msg_filter,
                )

                self.assertFalse(alpaka_filter2(parsed_row), f"{row}")
                self.assertEqual(reason_msg_filter.getvalue(), expected_error_msg, f"{row}")
