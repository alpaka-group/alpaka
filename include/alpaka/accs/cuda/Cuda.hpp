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

#include <alpaka/accs/cuda/Acc.hpp>                 // AccGpuCuda
#include <alpaka/accs/cuda/Atomic.hpp>              // AtomicCuda
#include <alpaka/accs/cuda/Common.hpp>              // ALPAKA_CUDA_RT_CHECK
#include <alpaka/accs/cuda/Dev.hpp>                 // Devices
#include <alpaka/accs/cuda/Event.hpp>               // EventCuda
#include <alpaka/accs/cuda/Exec.hpp>                // ExecGpuCuda
#include <alpaka/accs/cuda/Idx.hpp>                 // IdxCuda
#include <alpaka/accs/cuda/Mem.hpp>                 // Copy
#include <alpaka/accs/cuda/Rand.hpp>                // rand
#include <alpaka/accs/cuda/Stream.hpp>              // StreamCuda
#include <alpaka/accs/cuda/StreamEventTraits.hpp>   // StreamCuda & EventCuda
#include <alpaka/accs/cuda/WorkDiv.hpp>             // WorkDivCuda
