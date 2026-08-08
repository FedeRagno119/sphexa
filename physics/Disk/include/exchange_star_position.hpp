//
// Created by Noah Kubli on 04.03.2024.
//

#pragma once

#include "binary_energy.hpp"
#include "buffer_reduce.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"

namespace disk
{

//! @brief Compute the new star position by exchanging the force between the nodes and integrating the acceleration
template<typename StarData>
void computeAndExchangeStarPosition(StarData& star, double dt, double dt_m1)
{
    if (star.fixed_star == 1) { return; }

    const auto global_disk_force = buffer::mpiAllreduceSum(star.force_local);    //FR: sum force across all ranks
    const auto global_force = global_disk_force + star.force_binary;

    //FR: store potential and accelerations of star
    star.potential = global_disk_force[0];
    double a_starx = global_force[1] / star.m;
    double a_stary = global_force[2] / star.m;
    double a_starz = global_force[3] / star.m;

    //FR: function integrating one star coordinate and returning displacement - uses leapfrog method
    // a: acceleration
    // x_m1: displacement at previous timestep
    auto integrate = [dt, dt_m1](double a, double x_m1)
    {
        double deltaB = 0.5 * (dt + dt_m1);         //FR: 1/2 (dt_n + dt_n-1)
        auto   Val    = x_m1 * (1. / dt_m1);        //FR: v_n-1/2 = x_n-1 / dt_n-1
        auto   dx     = dt * Val + a * deltaB * dt; //FR: dx = v_(n-1/2)dt + a * 1/2 (dt_n + dt_n-1) * dt -> UPDATE SCHEME
        return dx;
    };

    //FR: evaluate displacements and update positions
    double dx = integrate(a_starx, star.position_m1[0]);
    double dy = integrate(a_stary, star.position_m1[1]);
    double dz = integrate(a_starz, star.position_m1[2]);
    star.position[0] += dx;
    star.position[1] += dy;
    star.position[2] += dz;
    star.position_m1[0] = dx;
    star.position_m1[1] = dy;
    star.position_m1[2] = dz;
}

/*! @brief Allreduce disk forces for both stars, compute energies (before integration so that
 *         position_m1 = dx_n gives v_{n+1/2}), then integrate both star positions.
 *
 *  Replaces the pair of computeAndExchangeStarPosition calls + computeBinaryEnergy call
 *  in the binary propagator, ensuring disk and star energies are consistent in time.
 */
template<typename Dataset, typename StarData>
void computeAndExchangeBinaryPositions(StarData& star1, StarData& star2, Dataset& d)
{
    const double dt    = d.minDt;
    const double dt_m1 = d.minDt_m1;

    // Step 1: allreduce disk forces for both stars
    const auto disk_force_1   = buffer::mpiAllreduceSum(star1.force_local);
    const auto disk_force_2   = buffer::mpiAllreduceSum(star2.force_local);
    const auto total_force_1  = disk_force_1 + star1.force_binary;
    const auto total_force_2  = disk_force_2 + star2.force_binary;
    if (!star1.fixed_star) { star1.potential = disk_force_1[0]; }
    if (!star2.fixed_star) { star2.potential = disk_force_2[0]; }

    //FR: writes star.ecin; star.egrav; star.etot
    computeBinaryEnergy(star1, star2, d);

    // Step 3: integrate positions
    auto integrate = [dt, dt_m1](double a, double x_m1)
    {
        double deltaB = 0.5 * (dt + dt_m1);
        auto   Val    = x_m1 * (1. / dt_m1);
        auto   dx     = dt * Val + a * deltaB * dt;
        return dx;
    };

    auto integrateStar = [&](StarData& star, const cstone::Vec4<double>& total_force)
    {
        // Captured even for a fixed star: the disk+companion force the star *would* feel is still
        // the physically meaningful quantity for torque diagnostics, even though its position isn't
        // integrated below.
        double ax = total_force[1] / star.m;
        double ay = total_force[2] / star.m;
        double az = total_force[3] / star.m;
        star.acc[0] = ax;
        star.acc[1] = ay;
        star.acc[2] = az;

        if (star.fixed_star == 1) { return; }

        double dx = integrate(ax, star.position_m1[0]);
        double dy = integrate(ay, star.position_m1[1]);
        double dz = integrate(az, star.position_m1[2]);

        star.position[0] += dx; star.position[1] += dy; star.position[2] += dz;
        star.position_m1[0] = dx; star.position_m1[1] = dy; star.position_m1[2] = dz;
    };

    integrateStar(star1, total_force_1);
    integrateStar(star2, total_force_2);
}

} // namespace disk
