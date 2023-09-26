#!/bin/bash
#
# SPDX-FileCopyrightText: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MPL-2.0

#-------------------------------------------------------------------------------
# -e: exit as soon as one command returns a non-zero exit code
# -o pipefail: pipeline returns exit code of the rightmost command with a non-zero exit code
# -u: treat unset variables as an error
# -v: Print shell input lines as they are read
# -x: Print command traces before executing command
set -eouvx pipefail
