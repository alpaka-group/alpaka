"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Verify generated combinations.
"""

from typing import Dict, Callable
import bashi


def verify(
    combination_list: bashi.CombinationList,
    param_value_matrix: bashi.ParameterValueMatrix,
    run_infos: Dict[str, Callable[..., bool]],
) -> bool:
    """Check if all expected parameter-value-pairs exists in the combination-list.

    Args:
        combination_list (CombinationList): The generated combination list.
        param_value_matrix (ParameterValueMatrix): The expected parameter-values-pairs are generated
            from the parameter-value-list.

    Returns:
        bool: True if it found all pairs
    """

    expected_param_val_tuple, unexpected_param_val_tuple = (
        bashi.get_expected_bashi_parameter_value_pairs(param_value_matrix, run_infos)
    )

    return bashi.check_parameter_value_pair_in_combination_list(
        combination_list, expected_param_val_tuple
    ) and bashi.check_unexpected_parameter_value_pair_in_combination_list(
        combination_list, unexpected_param_val_tuple
    )
