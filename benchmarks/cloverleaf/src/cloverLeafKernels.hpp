#pragma once

#include <alpaka/alpaka.hpp>

#include <experimental/mdspan>

using Data = float;
using Dim2 = alpaka::DimInt<2u>;
using Idx = std::uint32_t;

const Idx nx = 960; // Number of cells in x direction
const Idx ny = 960; // Number of cells in y direction

// Kernel to update the halo regions
struct UpdateHaloKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan density) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < nx && j < ny)
        {
            // Update halo cells for density (simplified example)
            // Assuming a single layer halo, and periodic boundary conditions
            if(i == 0)
                density(i, j) = density(nx - 2, j);
            if(i == nx - 1)
                density(i, j) = density(1, j);

            if(j == 0)
                density(i, j) = density(i, ny - 2);
            if(j == ny - 1)
                density(i, j) = density(i, 1);
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

        if(i < nx && j < ny)
        {
            pressure(i, j) = (gamma - 1.0f) * density(i, j) * energy(i, j);
            soundspeed(i, j) = sqrt(gamma * pressure(i, j) / density(i, j));
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
        MdSpan velocityY) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < nx && j < ny)
        {
            if(i >= static_cast<Idx>(0.0 * nx) && i < static_cast<Idx>(5.0 * nx / 10.0)
               && j >= static_cast<Idx>(0.0 * ny) && j < static_cast<Idx>(2.0 * ny / 10.0))
            {
                density(i, j) = 1.0f;
                energy(i, j) = 2.5f;
            }
            else
            {
                density(i, j) = 0.2f;
                energy(i, j) = 1.0f;
            }

            pressure(i, j) = 1.0f;
            velocityX(i, j) = 0.0f;
            velocityY(i, j) = 0.0f;
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
        float gamma) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < nx && j < ny)
        {
            // Compute pressure using ideal gas law: P = (gamma - 1) * density * energy
            pressure(i, j) = (gamma - 1.0f) * density(i, j) * energy(i, j);

            // Additional calculations to update velocities (this is a simplified example)
            velocityX(i, j) += pressure(i, j) * 0.1f;
            velocityY(i, j) += pressure(i, j) * 0.1f;
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
        MdSpan fluxDensity,
        MdSpan fluxEnergy,
        MdSpan fluxVelocityX,
        MdSpan fluxVelocityY) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < nx && j < ny)
        {
            // Compute fluxes (this is a simplified example)
            fluxDensity(i, j) = density(i, j) * velocityX(i, j);
            fluxEnergy(i, j) = energy(i, j) * velocityX(i, j);
            fluxVelocityX(i, j) = velocityX(i, j) * velocityX(i, j) + pressure(i, j);
            fluxVelocityY(i, j) = velocityY(i, j) * velocityX(i, j);
        }
    }
};

// Kernel for the advection step
struct AdvectionKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan density, MdSpan velocityX, MdSpan velocityY) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < nx && j < ny)
        {
            // Simple advection calculation (this is a simplified example)
            density(i, j) += (velocityX(i, j) + velocityY(i, j)) * 0.01f;
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
        MdSpan fluxDensity,
        MdSpan fluxEnergy,
        MdSpan fluxVelocityX,
        MdSpan fluxVelocityY) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < nx && j < ny)
        {
            // Update the cell-centered variables based on flux calculations
            density(i, j) -= fluxDensity(i, j) * 0.1f;
            energy(i, j) -= fluxEnergy(i, j) * 0.1f;
            velocityX(i, j) -= fluxVelocityX(i, j) * 0.1f;
            velocityY(i, j) -= fluxVelocityY(i, j) * 0.1f;
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
        MdSpan pressure,
        MdSpan viscosity) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1)
        {
            float gradVx = (velocityX(i + 1, j) - velocityX(i - 1, j)) * 0.5f;
            float gradVy = (velocityY(i, j + 1) - velocityY(i, j - 1)) * 0.5f;

            viscosity(i, j) = density(i, j) * (gradVx * gradVx + gradVy * gradVy) * 0.01f;

            // Apply viscosity to pressure
            pressure(i, j) += viscosity(i, j);
        }
    }
};

// Kernel to find the maximum velocity
struct MaxVelocityKernel
{
    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan velocityX, MdSpan velocityY, Data* maxVelocity) const -> void
    {
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < nx && j < ny)
        {
            float vx = velocityX(i, j);
            float vy = velocityY(i, j);
            float v = alpaka::math::sqrt(acc, (vx * vx + vy * vy));

            // Atomic operation to find the maximum velocity
            float val = alpaka::atomicMax(acc, maxVelocity, v);
            maxVelocity[0] = val;
        }
    }
};

[[maybe_unused]] static float calculateTimeStep(float dx, float dy, float maxVelocity, float cMax)
{
    // Compute the smallest grid spacing
    float minDx = std::min({dx, dy});

    // Calculate the time step based on the CFL condition
    float dt = cMax * minDx / maxVelocity;

    return dt;
}
