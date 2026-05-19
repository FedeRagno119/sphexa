//
// Created by Noah Kubli on 04.03.2024.
//

#pragma once

#include <algorithm>
#include <cmath>

#include "cstone/tree/definitions.h"
#include "central_force_gpu.hpp"
#include "get_ptr.hpp"
#include "central_potential.hpp"

namespace disk
{

template<typename Dataset, typename StarData>
void computeCentralForceImpl(size_t first, size_t last, Dataset& d, StarData& star)
{
    /* FR: "force accumulator"
    4-component vector storing: 
        (gravitational potential energy, 
        x-component force on star, 
        y-component force on star, 
        z-component force on star) 
    Called "force accumulator" since it accumulates the contributions from each particle on the star
    */
    cstone::Vec4<double>       force_local{};
    
    float                      t_star{std::numeric_limits<float>::infinity()};  //FR: stores minimum timestep constraint from central gravity
    const double               grav_softening2 = star.grav_softening * star.grav_softening;

    /*FR:
    Structure containing vectors of pointers to the data needed for the force evaluation, includes:
        particle positions,
        particle masses,
        particle accelerations,
        gravitational constant,
        star mass,
        softening radius,
        star position
    */
    const CentralPotentialData data{d.x.data(),  d.y.data(), d.z.data(), d.m.data(),  d.ax.data(), d.ay.data(),
                                    d.az.data(), d.g,        star.m,     grav_softening2, 1.0,      star.position};

#pragma omp declare reduction(add_force : cstone::Vec4<double> : omp_out = omp_out + omp_in) initializer(omp_priv = {})

#pragma omp parallel for reduction(add_force : force_local) reduction(min : t_star)
    for (size_t i = first; i < last; i++) //FR: loop over particles in rank
    {
        if (star.potentialType == StarPotentialType::newtonian) 
        { 
            newtonianGravity(data, i, force_local, t_star); 
        }
        else if (star.potentialType == StarPotentialType::einstein_precession)
        {
            einsteinPrecession(data, i, force_local, t_star);
        }
    }

    star.force_local = force_local;
    star.t_star      = t_star;
}

template<typename Dataset, typename StarData>
void computeCentralForce(size_t startIndex, size_t endIndex, Dataset& d, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeCentralForceGPU(startIndex, endIndex, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d), getPtr<"ax">(d),
                               getPtr<"ay">(d), getPtr<"az">(d), getPtr<"m">(d), d.g, star);
    }
    else { computeCentralForceImpl(startIndex, endIndex, d, star); }
}

template<typename Dataset, typename StarData>
void computeBinaryCentralForceImpl(size_t first, size_t last, Dataset& d, StarData& star1, StarData& star2)
{
    cstone::Vec4<double> force1_local{};
    cstone::Vec4<double> force2_local{};
    float                t_star1{std::numeric_limits<float>::infinity()};
    float                t_star2{std::numeric_limits<float>::infinity()};

    const double grav_softening2_1 = star1.grav_softening * star1.grav_softening;
    const double grav_softening2_2 = star2.grav_softening * star2.grav_softening;

    const CentralPotentialData data1{d.x.data(), d.y.data(), d.z.data(), d.m.data(), d.ax.data(), d.ay.data(),
                                     d.az.data(), d.g, star1.m, grav_softening2_1, 1.0, star1.position};
    const CentralPotentialData data2{d.x.data(), d.y.data(), d.z.data(), d.m.data(), d.ax.data(), d.ay.data(),
                                     d.az.data(), d.g, star2.m, grav_softening2_2, 1.0, star2.position};

#pragma omp declare reduction(add_force : cstone::Vec4<double> : omp_out = omp_out + omp_in) initializer(omp_priv = {})

#pragma omp parallel for reduction(add_force : force1_local, force2_local) reduction(min : t_star1, t_star2)
    for (size_t i = first; i < last; i++)
    {
        if (star1.potentialType == StarPotentialType::newtonian)
        {
            newtonianGravity(data1, i, force1_local, t_star1);
            newtonianGravity(data2, i, force2_local, t_star2);
        }
        else if (star1.potentialType == StarPotentialType::einstein_precession)
        {
            einsteinPrecession(data1, i, force1_local, t_star1);
            einsteinPrecession(data2, i, force2_local, t_star2);
        }
    }

    star1.force_local = force1_local;
    star1.t_star      = t_star1;
    star2.force_local = force2_local;
    star2.t_star      = t_star2;
}

template<typename Dataset, typename StarData>
void computeBinaryCentralForce(size_t startIndex, size_t endIndex, Dataset& d, StarData& star1, StarData& star2)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeBinaryCentralForceGPU(startIndex, endIndex, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d),
                                     getPtr<"ax">(d), getPtr<"ay">(d), getPtr<"az">(d), getPtr<"m">(d), d.g,
                                     star1, star2);
    }
    else { computeBinaryCentralForceImpl(startIndex, endIndex, d, star1, star2); }
}

} // namespace disk
