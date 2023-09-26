#!/bin/bash
#
# SPDX-FileCopyrightText: Travis CI GmbH <contact+travis-build@travis-ci.org>
# SPDX-FileCopyrightText: Helmholtz-Zentrum Dresden-Rossendorf e.V. <https://www.hzdr.de>
# SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
#
# SPDX-FileContributor: Travis CI GmbH <contact+travis-build@travis-ci.org>
# SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
# SPDX-FileContributor: René Widera <r.widera@hzdr.de>
# SPDX-FileContributor: Benjamin Worpitz <benjaminworpitz@gmail.com>
#
# SPDX-License-Identifier: MIT

ANSI_RED="\033[31m"
ANSI_RESET="\033[0m"

travis_retry() {
  set +euo pipefail
  local result=0
  local count=1
  local max=666
  while [ $count -le $max ]; do
    [ $result -ne 0 ] && {
      echo -e "\n${ANSI_RED}The command \"$*\" failed. Retrying, $count of $max.${ANSI_RESET}\n" >&2
    }
    "$@"
    result=$?
    [ $result -eq 0 ] && break
    count=$((count + 1))
    sleep 1
  done
  [ $count -gt $max ] && {
    echo -e "\n${ANSI_RED}The command \"$*\" failed $max times.${ANSI_RESET}\n" >&2
  }
  return $result
}
