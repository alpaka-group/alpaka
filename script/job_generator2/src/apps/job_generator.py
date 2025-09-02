"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate CI jobs for alpaka.
"""

import sys
import bashi
import alpaka_bashi


def setup_row_printer() -> None:
    """Set extra configurations for the bashi.print_row_nice() function"""
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.BUILD_TYPE, "buildType")
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.JOB_EXECUTION_TYPE, "jobType")

    for val_name, aliases in alpaka_bashi.get_version_aliases().items():
        bashi.add_print_row_nice_version_alias(val_name, aliases)


def main() -> None:
    """Entry point"""
    setup_row_printer()

    param_matrix: bashi.ParameterValueMatrix = bashi.get_parameter_value_matrix(
        software_versions=alpaka_bashi.get_alpaka_version()
    )

    alpaka_filter = alpaka_bashi.AlpakaFilter()
    runtime_infos = bashi.get_runtime_infos(param_matrix)
    runtime_infos |= alpaka_bashi.get_runtime_infos()

    comb_list: bashi.CombinationList = bashi.generate_combination_list(
        parameter_value_matrix=param_matrix,
        runtime_infos=runtime_infos,
        custom_filter=alpaka_filter,
        # change me to display which combinations passed and did not pass the filter chain
        debug_print=bashi.FilterDebugMode.OFF,
    )

    print(f"number of combinations: {len(comb_list)}", file=sys.stderr)

    alpaka_bashi.add_combinations_parameters(comb_list)

    if not alpaka_bashi.verify(comb_list, param_matrix, runtime_infos):
        print("ERROR: Result is incorrect", file=sys.stderr)
        sys.exit(1)

    # for c in comb_list:
    #     bashi.print_row_nice(c)

    print("Result is correct", file=sys.stderr)

    pipelines = alpaka_bashi.distribute_to_pipelines(comb_list)

    sys.exit(0)


if __name__ == "__main__":
    main()
