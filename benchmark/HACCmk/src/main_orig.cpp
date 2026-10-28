/* Copyright (c) 2020 , Argonne National Laboratory   (Zheming Jin)
 * Copyright (c) 2020-, Oak Ridge National Laboratory   (Zheming Jin)
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "helper.hpp"
#include "misc.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>

#if defined(_OPENMP)
#    include <omp.h>
#endif


// kernel signature kept identical to the original HACC benchmark
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template<typename T>
void haccmk(
    int const repeat,
    size_t const n, // global size
    int const ilp, // inner loop count
    T const fsrrmax,
    T const mpRsm,
    T const fcoeff,
    T const* __restrict xx,
    T const* __restrict yy,
    T const* __restrict zz,
    T const* __restrict mass,
    T* __restrict vx2,
    T* __restrict vy2,
    T* __restrict vz2)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    float totalTime = 0.f;

    for(int r = 0; r < repeat; r++)
    {
        auto start = std::chrono::steady_clock::now();

#if defined(_OPENMP)
#    pragma omp parallel for
#endif
        for(int i = 0; i < n; i++)
        {
            float const ma0 = 0.269327f;
            float const ma1 = -0.0750978f;
            float const ma2 = 0.0114808f;
            float const ma3 = -0.00109313f;
            float const ma4 = 0.0000605491f;
            float const ma5 = -0.00000147177f;

            float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

            xi = 0.f;
            yi = 0.f;
            zi = 0.f;

            float xxi = xx[i];
            float yyi = yy[i];
            float zzi = zz[i];

            for(int j = 0; j < ilp; j++)
            {
                dxc = xx[j] - xxi;
                dyc = yy[j] - yyi;
                dzc = zz[j] - zzi;

                r2 = dxc * dxc + dyc * dyc + dzc * dzc;

                m = mass[j] * (r2 < fsrrmax);

                f = r2 + mpRsm;
                f = m * (1.f / (f * sqrtf(f)) - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));

                xi = xi + f * dxc;
                yi = yi + f * dyc;
                zi = zi + f * dzc;
            }

            /* No `+=` is used.
             * This differs to the original publication but is necessary due to the OpenMP bug.
             */
            vx2[i] = xi * fcoeff;
            vy2[i] = yi * fcoeff;
            vz2[i] = zi * fcoeff;
        }

        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        // first round is a warmup round
        if(r != 0)
            totalTime += static_cast<float>(time);
    }
    printf("Average kernel execution time %f (s)\n", (totalTime * 1e-9f) / static_cast<float>(repeat - 1));
}

int main(int argc, char* argv[])
{
    int n2 = 1;
    int n1 = 1;
    int repeat = 1;

    if(int const ret = hacc::parseCmd(argc, argv, n2, n1, repeat))
        return ret;

    float fsrrmax2, mpRsm2, fcoeff, dx1, dy1, dz1, dx2, dy2, dz2;
    int i;

    printf("Outer loop count is set %d\n", n1);
    printf("Inner loop count is set %d\n", n2);

    float* xx = hacc::allocAligned<float>(n2);
    float* yy = hacc::allocAligned<float>(n2);
    float* zz = hacc::allocAligned<float>(n2);
    float* mass = hacc::allocAligned<float>(n2);
    float* vx2 = hacc::allocAligned<float>(n2);
    float* vy2 = hacc::allocAligned<float>(n2);
    float* vz2 = hacc::allocAligned<float>(n2);
    float* vx2Hw = hacc::allocAligned<float>(n2);
    float* vy2Hw = hacc::allocAligned<float>(n2);
    float* vz2Hw = hacc::allocAligned<float>(n2);

    /* Initial data preparation */
    fcoeff = 0.23f;
    fsrrmax2 = 0.5f;
    mpRsm2 = 0.03f;
    dx1 = 1.0f / static_cast<float>(n2);
    dy1 = 2.0f / static_cast<float>(n2);
    dz1 = 3.0f / static_cast<float>(n2);
    xx[0] = 0.f;
    yy[0] = 0.f;
    zz[0] = 0.f;
    mass[0] = 2.f;

    for(i = 1; i < n2; i++)
    {
        xx[i] = xx[i - 1] + dx1;
        yy[i] = yy[i - 1] + dy1;
        zz[i] = zz[i - 1] + dz1;
        mass[i] = static_cast<float>(i) * 0.01f + xx[i];
    }

    for(i = 0; i < n2; i++)
    {
        vx2[i] = 0.f;
        vy2[i] = 0.f;
        vz2[i] = 0.f;
        vx2Hw[i] = 0.f;
        vy2Hw[i] = 0.f;
        vz2Hw[i] = 0.f;
    }

    for(i = 0; i < n1; ++i)
    {
        hacc::haccmk_gold(n2, xx[i], yy[i], zz[i], fsrrmax2, mpRsm2, xx, yy, zz, mass, &dx2, &dy2, &dz2);
        vx2[i] = vx2[i] + dx2 * fcoeff;
        vy2[i] = vy2[i] + dy2 * fcoeff;
        vz2[i] = vz2[i] + dz2 * fcoeff;
    }

    haccmk(repeat, n1, n2, fsrrmax2, mpRsm2, fcoeff, xx, yy, zz, mass, vx2Hw, vy2Hw, vz2Hw);

    int error = hacc::verify(n2, vx2, vy2, vz2, vx2Hw, vy2Hw, vz2Hw);

    hacc::freeAligned(xx);
    hacc::freeAligned(yy);
    hacc::freeAligned(zz);
    hacc::freeAligned(mass);
    hacc::freeAligned(vx2);
    hacc::freeAligned(vy2);
    hacc::freeAligned(vz2);
    hacc::freeAligned(vx2Hw);
    hacc::freeAligned(vy2Hw);
    hacc::freeAligned(vz2Hw);

    return error;
}
