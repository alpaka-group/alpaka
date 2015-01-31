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

#include <alpaka/core/BasicDims.hpp>
#include <alpaka/core/BasicExtents.hpp>
#include <alpaka/core/BasicWorkDiv.hpp>
#include <alpaka/core/BasicMem.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/EnabledAccelerators.hpp>
#include <alpaka/core/KernelExecCreator.hpp>
#include <alpaka/core/MemBufPlainPtrWrapper.hpp>
#include <alpaka/core/Ops.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Vec.hpp>
#include <alpaka/core/WorkDivHelpers.hpp>

#include <alpaka/interfaces/IAcc.hpp>

#include <alpaka/host/Mem.hpp>
#include <alpaka/host/MemSpace.hpp>
#include <alpaka/host/SysInfo.hpp>

#include <alpaka/traits/Acc.hpp>
#include <alpaka/traits/Atomic.hpp>
#include <alpaka/traits/BlockSharedExternMemSizeBytes.hpp>
#include <alpaka/traits/Device.hpp>
#include <alpaka/traits/Dim.hpp>
#include <alpaka/traits/Event.hpp>
#include <alpaka/traits/Extents.hpp>
#include <alpaka/traits/Idx.hpp>
#include <alpaka/traits/Mem.hpp>
#include <alpaka/traits/Stream.hpp>
#include <alpaka/traits/Wait.hpp>
#include <alpaka/traits/WorkDiv.hpp>
