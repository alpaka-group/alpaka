# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest

from bashi.globals import CLANG_CUDA, CMAKE, DEVICE_COMPILER, GCC, HOST_COMPILER
from bashi.types import ParameterValuePair
from utils import default_remove_test, parse_expected_val_pairs

from alpaka_bashi.verify import remove_unsupported_cmake_versions_for_clang_host_compiler


class TestUnsupportedCmakeVersionsForClangHostCompiler(unittest.TestCase):
    def test_remove_invalid_combinations(self):
        test_param_value_pairs: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, CLANG_CUDA, 22), (CMAKE, "3.30.3")),
                ((HOST_COMPILER, CLANG_CUDA, 23), (CMAKE, "3.30.4")),
                ((DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "3.27.1")),
                ((DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "3.30.5")),
                ((DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "3.31.1")),
                ((DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "4.5.2")),
                ((HOST_COMPILER, CLANG_CUDA, 24), (CMAKE, "3.30.1")),
                ((HOST_COMPILER, CLANG_CUDA, 24), (CMAKE, "3.31.0")),
                ((HOST_COMPILER, CLANG_CUDA, 27), (CMAKE, "4.6.1")),
            ]
        )

        expected_results: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, CLANG_CUDA, 22), (CMAKE, "3.30.3")),
                ((DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "3.31.1")),
                ((DEVICE_COMPILER, CLANG_CUDA, 23), (CMAKE, "4.5.2")),
                ((HOST_COMPILER, CLANG_CUDA, 24), (CMAKE, "3.31.0")),
                ((HOST_COMPILER, CLANG_CUDA, 27), (CMAKE, "4.6.1")),
            ]
        )

        default_remove_test(
            remove_unsupported_cmake_versions_for_clang_host_compiler,
            test_param_value_pairs,
            expected_results,
            self,
        )
