/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/accs/omp/omp2/blocks/Acc.hpp>  // AccCpuOmp2Blocks
#include <alpaka/accs/omp/omp2/blocks/Exec.hpp> // ExecCpuOmp2Blocks
#include <alpaka/accs/omp/Idx.hpp>              // IdxOmp
#include <alpaka/accs/omp/Atomic.hpp>           // AtomicOmp
#include <alpaka/devs/cpu/Mem.hpp>              // Copy
#include <alpaka/devs/cpu/Rand.hpp>             // rand
