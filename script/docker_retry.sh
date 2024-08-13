#!/bin/bash
#
# Copyright 2019-2020 Benjamin Worpitz, Rene Widera
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

# rerun docker command if error 125 (
#   - triggered by image download problems
#   - wait 30 seconds before retry
docker_retry() {
  # apply `set +euo pipefail` in a local scope so that the following script is not affected and 
  # e.g. exit on failure is not deactivated
  (
    set +euo pipefail
    local result=0
    local count=1
    while [ $count -le 3 ]; do
      [ $result -eq 125 ] && {
      echo_red "\nThe command \"$*\" failed. Retrying, $count of 3.\n" >&2
      }
      "$@"
      result=$?
      [ $result -ne 125 ] && break
      count=$((count + 1))
      sleep 30
    done
    [ $count -gt 3 ] && {
      echo_red "\nThe command \"$*\" failed 3 times.\n" >&2
    }
    return $result
  )
}
