"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

from typing import Dict, Callable, IO
import bashi


class AlpakaFilter(bashi.FilterBase):
    """Alpaka specific filter rules."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        output: IO[bashi.Parameter] | None = None,
    ):
        super().__init__(runtime_infos, output)

    def __call__(
        self,
        row: bashi.ParameterValueTuple,
    ) -> bool:
        """Check if given parameter-value-tuple is valid

        Args:
            row (ParameterValueTuple): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """

        return True
