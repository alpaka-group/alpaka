/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License
*
*
* You should have received a copy of the GNU Lesser General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/openmp/AccOpenMpFwd.hpp>   // AccOpenMp

#include <alpaka/host/Event.hpp>            // EventHost

namespace alpaka
{
    namespace event
    {
        //#############################################################################
        //! The OpenMP accelerator event.
        //#############################################################################
        template<>
        class Event<AccOpenMp> :
            public alpaka::host::detail::EventHost
        {
        public:
            using TAcc = AccOpenMp;
        };
    }
}
