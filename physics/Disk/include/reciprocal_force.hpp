//
// Created by Federico Ragno on 11.03.2026.
//
// Function containing computation of star-star forces
//
#pragma once

#define CONST_PI 3.14159265358979323846

#include <cassert>
#include <cmath>
#include "cstone/primitives/stl.hpp"
#include "cstone/tree/definitions.h"
#include "binary_data.hpp"

namespace disk 
{
    template <typename Dataset, typename StarData>
    void newtonianBinaryForce(const StarData& star1, const StarData& star2, cstone::Vec4<double>& force_binary_1, cstone::Vec4<double>& force_binary_2, float& t_binary, Dataset& d) 
    {
        const double dx = star2.position[0] - star1.position[0];
        const double dy = star2.position[1] - star1.position[1];
        const double dz = star2.position[2] - star1.position[2];
        const double min_dist = star1.inner_size + star2.inner_size;
        const double dist2 = stl::max(min_dist*min_dist, dx * dx + dy * dy + dz * dz);
        const double dist  = std::sqrt(dist2);
        const double dist3 = dist2 * dist;

        const double f_strength = (1. / dist3) * star1.m * star2.m * d.g;

        force_binary_1[0] = - (d.g * star2.m * star1.m) / dist;
        force_binary_1[1] = f_strength * dx;
        force_binary_1[2] = f_strength * dy;
        force_binary_1[3] = f_strength * dz;
        
        force_binary_2[0] = force_binary_1[0];
        force_binary_2[1] = - force_binary_1[1];
        force_binary_2[2] = - force_binary_1[2];
        force_binary_2[3] = - force_binary_1[3];

        const double t_binary_sq = (4 * CONST_PI * CONST_PI * dist3) / (d.g * (star1.m + star2.m));
        t_binary = std::sqrt(t_binary_sq);
    }

    template <typename Dataset, typename StarData>
    void computeBinaryForce(StarData& star1, StarData& star2, Dataset& d){
        /*
        IMPORTANT NOTES:
            1) must update force_binary of both star objects
                IMPORTANT: must NOT be additive (+=) but calcualted on the spot
            2) must differentiate between newtonian and eisteinian potential
            3) (MAYBE) must store minimum timestep due to binary orbitals
        */

        cstone::Vec4<double> force_binary_1{};
        cstone::Vec4<double> force_binary_2{};

        float t_binary{std::numeric_limits<float>::infinity()};
        
        assert(binary_is_consistent(star1, star2) && "Assertion error in computing binary forces: stars are not consistent");

        if (star1.potentialType == StarPotentialType::newtonian) 
        { 
            newtonianBinaryForce(star1, star2, force_binary_1, force_binary_2, t_binary, d); 
        }
        else if (star1.potentialType == StarPotentialType::einstein_precession)
        {
            //einsteinianBinaryForce();
        }

        star1.force_binary = force_binary_1;
        star2.force_binary = force_binary_2;    

        star1.t_binary = t_binary;
        star2.t_binary = t_binary;
    }

} // namespace disk