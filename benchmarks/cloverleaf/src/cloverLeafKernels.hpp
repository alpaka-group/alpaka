#pragma once

#include <alpaka/alpaka.hpp>

#include <experimental/mdspan>

using Data = float;
using Dim3 = alpaka::DimInt<3u>;
using Idx = std::uint32_t;

const Idx nx = 512; // Number of cells in x direction
const Idx ny = 512; // Number of cells in y direction
const Idx nz = 512; // Number of cells in z direction

// Kernel to update the halo regions
struct UpdateHaloKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan density) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            // Update halo cells for density (simplified example)
            // Assuming a single layer halo, and periodic boundary conditions
            if(i == 0)
                density(i, j, k) = density(nx - 2, j, k);
            if(i == nx - 1)
                density(i, j, k) = density(1, j, k);

            if(j == 0)
                density(i, j, k) = density(i, ny - 2, k);
            if(j == ny - 1)
                density(i, j, k) = density(i, 1, k);

            if(k == 0)
                density(i, j, k) = density(i, j, nz - 2);
            if(k == nz - 1)
                density(i, j, k) = density(i, j, 1);
        }
    }
};

// Kernel to compute the ideal gas equation of state
struct IdealGasKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan density,
        MdSpan energy,
        MdSpan pressure,
        MdSpan soundspeed,
        float gamma) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            pressure(i, j, k) = (gamma - 1.0f) * density(i, j, k) * energy(i, j, k);
            soundspeed(i, j, k) = sqrt(gamma * pressure(i, j, k) / density(i, j, k));
        }
    }
};

// Kernel to initialize the simulation variables
struct InitializerKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan density,
        MdSpan energy,
        MdSpan pressure,
        MdSpan velocityX,
        MdSpan velocityY,
        MdSpan velocityZ) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            density(i, j, k) = 1.0f; // Initial density
            energy(i, j, k) = 2.5f; // Initial energy
            pressure(i, j, k) = 1.0f; // Initial pressure
            velocityX(i, j, k) = 0.0f; // Initial velocity

            if(i < nx && j < ny && k < nz)
            {
                // Simple advection calculation (this is a simplified example)
                density(i, j, k) += (velocityX(i, j, k) + velocityY(i, j, k) + velocityZ(i, j, k)) * 0.01f;
            }
        }
    }
};

// Kernel to compute the equation of state (EOS) and additional calculations
struct EOSKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan density,
        MdSpan energy,
        MdSpan pressure,
        MdSpan velocityX,
        MdSpan velocityY,
        MdSpan velocityZ,
        float gamma) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            // Compute pressure using ideal gas law: P = (gamma - 1) * density * energy
            pressure(i, j, k) = (gamma - 1.0f) * density(i, j, k) * energy(i, j, k);

            // Additional calculations to update velocities (this is a simplified example)
            velocityX(i, j, k) += pressure(i, j, k) * 0.1f;
            velocityY(i, j, k) += pressure(i, j, k) * 0.1f;
            velocityZ(i, j, k) += pressure(i, j, k) * 0.1f;
        }
    }
};

// Kernel for Flux calculations
struct FluxKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan density,
        MdSpan energy,
        MdSpan pressure,
        MdSpan velocityX,
        MdSpan velocityY,
        MdSpan velocityZ,
        MdSpan fluxDensity,
        MdSpan fluxEnergy,
        MdSpan fluxVelocityX,
        MdSpan fluxVelocityY,
        MdSpan fluxVelocityZ) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            // Compute fluxes (this is a simplified example)
            fluxDensity(i, j, k) = density(i, j, k) * velocityX(i, j, k);
            fluxEnergy(i, j, k) = energy(i, j, k) * velocityX(i, j, k);
            fluxVelocityX(i, j, k) = velocityX(i, j, k) * velocityX(i, j, k) + pressure(i, j, k);
            fluxVelocityY(i, j, k) = velocityY(i, j, k) * velocityX(i, j, k);
            fluxVelocityZ(i, j, k) = velocityZ(i, j, k) * velocityX(i, j, k);
        }
    }
};

// Kernel for the advection step
struct AdvectionKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan density,
        MdSpan velocityX,
        MdSpan velocityY,
        MdSpan velocityZ) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            // Simple advection calculation (this is a simplified example)
            density(i, j, k) += (velocityX(i, j, k) + velocityY(i, j, k) + velocityZ(i, j, k)) * 0.01f;
        }
    }
};

// Kernel for the Lagrangian step
struct LagrangianKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan density,
        MdSpan energy,
        MdSpan velocityX,
        MdSpan velocityY,
        MdSpan velocityZ,
        MdSpan fluxDensity,
        MdSpan fluxEnergy,
        MdSpan fluxVelocityX,
        MdSpan fluxVelocityY,
        MdSpan fluxVelocityZ) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            // Update the cell-centered variables based on flux calculations
            density(i, j, k) -= fluxDensity(i, j, k) * 0.1f;
            energy(i, j, k) -= fluxEnergy(i, j, k) * 0.1f;
            velocityX(i, j, k) -= fluxVelocityX(i, j, k) * 0.1f;
            velocityY(i, j, k) -= fluxVelocityY(i, j, k) * 0.1f;
            velocityZ(i, j, k) -= fluxVelocityZ(i, j, k) * 0.1f;
        }
    }
};

// Kernel for viscosity calculations
struct ViscosityKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan density,
        MdSpan velocityX,
        MdSpan velocityY,
        MdSpan velocityZ,
        MdSpan pressure,
        MdSpan viscosity) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1)
        {
            float gradVx = (velocityX(i + 1, j, k) - velocityX(i - 1, j, k)) * 0.5f;
            float gradVy = (velocityY(i, j + 1, k) - velocityY(i, j - 1, k)) * 0.5f;
            float gradVz = (velocityZ(i, j, k + 1) - velocityZ(i, j, k - 1)) * 0.5f;

            viscosity(i, j, k) = density(i, j, k) * (gradVx * gradVx + gradVy * gradVy + gradVz * gradVz) * 0.01f;

            // Apply viscosity to pressure
            pressure(i, j, k) += viscosity(i, j, k);
        }
    }
};

// Kernel to find the maximum velocity
struct MaxVelocityKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        MdSpan velocityX,
        MdSpan velocityY,
        MdSpan velocityZ,
        Data* maxVelocity) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];

        if(i < nx && j < ny && k < nz)
        {
            float vx = velocityX(i, j, k);
            float vy = velocityY(i, j, k);
            float vz = velocityZ(i, j, k);
            float v = alpaka::math::sqrt(acc, (vx * vx + vy * vy + vz * vz));

            // Atomic operation to find the maximum velocity
            float val = alpaka::atomicMax(acc, maxVelocity, v);
            maxVelocity[0] = val;
        }
    }
};

[[maybe_unused]] static float calculateTimeStep(float dx, float dy, float dz, float maxVelocity, float cMax)
{
    // Compute the smallest grid spacing
    float minDx = std::min({dx, dy, dz});

    // Calculate the time step based on the CFL condition
    float dt = cMax * minDx / maxVelocity;

    return dt;
}
