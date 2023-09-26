/*
 * SPDX-FileCopyrightText: University of Bristol <https://www.bristol.ac.uk>
 * SPDX-FileCopyrightText: Georgia Institute of Technology <https://www.gatech.edu>
 * SPDX-FileCopyrightText: Organisation européenne pour la recherche nucléaire (CERN) <https://www.cern.ch>
 *
 * SPDX-FileContributor: Tom Deakin <tom.deakin@bristol.ac.uk>
 * SPDX-FileContributor: Simon McIntosh-Smith <s.mcintosh-smith@bristol.ac.uk>
 * SPDX-FileContributor: Jeffrey Young <jyoung9@gatech.edu>
 * SPDX-FileContributor: Bernhard Manfred Gruber <bernhardmgruber@gmail.com>
 *
 * SPDX-License-Identifier: LicenseRef-Babelstream
 */

#pragma once

#include "Stream.h"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <vector>

inline constexpr auto IMPLEMENTATION_STRING = "alpaka";

using Dim = alpaka::DimInt<1>;
using Idx = int;
using Vec = alpaka::Vec<Dim, Idx>;
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

template<typename T>
struct AlpakaStream : Stream<T>
{
    AlpakaStream(Idx arraySize, Idx deviceIndex);

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

    using PlatformHost = alpaka::PlatformCpu;
    using DevHost = alpaka::Dev<PlatformHost>;
    using PlatformAcc = alpaka::Platform<Acc>;
    using DevAcc = alpaka::Dev<Acc>;
    using BufHost = alpaka::Buf<alpaka::DevCpu, T, Dim, Idx>;
    using BufAcc = alpaka::Buf<Acc, T, Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

private:
    Idx arraySize;
    PlatformHost platformHost;
    DevHost devHost;
    PlatformAcc platformAcc;
    DevAcc devAcc;
    BufHost sums;
    BufAcc d_a;
    BufAcc d_b;
    BufAcc d_c;
    BufAcc d_sum;
    Queue queue;
};

void listDevices();
std::string getDeviceName(int deviceIndex);
std::string getDeviceDriver(int device);
