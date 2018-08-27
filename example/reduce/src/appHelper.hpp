/**
 * \file
 * Copyright 2018 Matthias Werner, Jonas Schenke
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
 *
 */

#include <ctime>
#include <iostream>

inline
void print_header(const char* app_name, unsigned n1, unsigned n2) {

  std::time_t now = std::time(nullptr);
  std::cout << "; "<< app_name
            << " " << n1
            << " " << n2
            << ", " << strtok(ctime(&now), "\n")
            << "\n";

  std::cout << "dev_id"
            << ",\tdev_name"
            << ",\t\t\tn"
            << ",\tnumSMs"
            << ",\tblocks_i"
            << ",\tblocks_i/numSMs"
            << ",\tblocks_n"
            << ",\tTBlocksize"
            << ",\tmin_time"
            << ",\tmax_throughput (in GB/s)"
            << "\n"
    ;

}
