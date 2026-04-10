//
// Created by Noah Kubli on 04.03.2024.
//

#pragma once

#include "buffer_reduce.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"

namespace disk
{

//! @brief Compute the new star position by exchanging the force between the nodes and integrating the acceleration
template<typename StarData>
void computeAndExchangeStarPosition(StarData& star, double dt, double dt_m1)
{
    if (star.fixed_star == 1) { return; }

    const auto global_force = buffer::mpiAllreduceSum(star.force_local) + star.force_binary;    //FR: sum force across all ranks
    
    //FR: store potential and accelerations of star
    star.potential = global_force[0];
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
} // namespace disk
