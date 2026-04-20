# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
from collections import OrderedDict
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from utils import parse_param_value_tuples
from alpaka_bashi.alpaka_filter import all_backends_fine


class TestAlpakaGetValidBackendCombinations(unittest.TestCase):
    def test_empty_row_empty_backend_list(self):
        self.assertTrue(all_backends_fine(OrderedDict(), [], []))

    def test_empty_row_single_backend(self):
        self.assertTrue(
            all_backends_fine(
                row=OrderedDict(),
                backends=[ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
                all_available_backends=[ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
            )
        )

    TEST_DATA_EMPTY_ROW_TWO_BACKEND = [
        (
            [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_GPU_CUDA_ENABLE],
            [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_GPU_CUDA_ENABLE],
        ),
        (
            [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_GPU_CUDA_ENABLE],
            [ALPAKA_ACC_GPU_CUDA_ENABLE],
        ),
        (
            [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
            [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_GPU_CUDA_ENABLE],
        ),
    ]

    def test_empty_row_two_backend(self):
        for enabled_backends, all_backends in self.TEST_DATA_EMPTY_ROW_TWO_BACKEND:
            with self.subTest(enabled_backends=enabled_backends, all_backends=all_backends):
                self.assertTrue(all_backends_fine(OrderedDict(), enabled_backends, all_backends))

    def test_single_row_empty_backend_list(self):
        self.assertTrue(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)]), [], []
            )
        )
        self.assertTrue(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)]),
                [],
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
            )
        )

    def test_single_row_single_backend_valid(self):
        self.assertTrue(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)]),
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
            )
        )

        self.assertTrue(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)]),
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
                [],
            )
        )

    def test_single_row_single_backend_invalid(self):
        self.assertFalse(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)]),
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
            )
        )
        self.assertFalse(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)]),
                [],
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE],
            )
        )

    def test_cuda_backend(self):
        self.assertFalse(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_GPU_CUDA_ENABLE, OFF)]),
                [ALPAKA_ACC_GPU_CUDA_ENABLE],
                [ALPAKA_ACC_GPU_CUDA_ENABLE],
            )
        )

        self.assertTrue(
            all_backends_fine(
                parse_param_value_tuples([(ALPAKA_ACC_GPU_CUDA_ENABLE, 11.4)]),
                [ALPAKA_ACC_GPU_CUDA_ENABLE],
                [ALPAKA_ACC_GPU_CUDA_ENABLE],
            )
        )

    TEST_DATA_MULTIPLE_ROW_MULTIPLE_BACKEND_VALID_CASE = [
        (
            parse_param_value_tuples(
                [
                    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                ]
            ),
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
        (
            parse_param_value_tuples(
                [
                    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                ]
            ),
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
        (
            parse_param_value_tuples(
                [
                    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                ]
            ),
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
        (
            parse_param_value_tuples(
                [
                    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                    (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                ]
            ),
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_CUDA_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
    ]

    def test_multiple_row_multiple_backend_valid(self):
        for row, backends, all_backends in self.TEST_DATA_MULTIPLE_ROW_MULTIPLE_BACKEND_VALID_CASE:
            with self.subTest(row=row, backends=backends, all_backends=all_backends):
                self.assertTrue(
                    all_backends_fine(row, backends, all_backends),
                    f"\nrow:\n  {row}\nbackends:\n  {backends}\nall backends:  \n{all_backends}",
                )

    TEST_DATA_MULTIPLE_ROW_MULTIPLE_BACKEND_INVALID_CASE = [
        (
            parse_param_value_tuples(
                [
                    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, OFF),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                ]
            ),
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
        (
            parse_param_value_tuples(
                [
                    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, OFF),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
                ]
            ),
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
        (
            parse_param_value_tuples(
                [
                    (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON),
                    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                    (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                    (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                ]
            ),
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
            [
                ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE,
                ALPAKA_ACC_GPU_HIP_ENABLE,
            ],
        ),
    ]

    def test_multiple_row_multiple_backend_invalid(self):
        for (
            row,
            backends,
            all_backends,
        ) in self.TEST_DATA_MULTIPLE_ROW_MULTIPLE_BACKEND_INVALID_CASE:
            with self.subTest(row=row, backends=backends, all_backends=all_backends):
                self.assertFalse(
                    all_backends_fine(row, backends, all_backends),
                    f"\nrow:\n  {row}\nbackends:\n  {backends}\nall backends:  \n{all_backends}",
                )
