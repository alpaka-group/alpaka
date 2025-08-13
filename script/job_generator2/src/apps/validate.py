"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Validate if a combination parameter-values is valid for alpaka.
"""

import sys
import bashiValidate
import alpaka_bashi


def main():
    """Entry point function."""

    # TODO: find solution to add arguments, which are not a version number or ON or OFF, e.g.
    # --build_type=Release
    validator = bashiValidate.Validator()
    validator.add_custom_filter(alpaka_bashi.AlpakaFilter())
    sys.exit(int(not validator.validate()))


if __name__ == "__main__":
    main()
