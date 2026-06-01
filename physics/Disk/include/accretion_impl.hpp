//
// Created by Noah Kubli on 14.03.2024.
//

#pragma once

#include "cstone/tree/definitions.h"
#include "removal_statistics.hpp"
#include "star_data.hpp"
#include <cmath>

namespace disk
{

template<typename Dataset>
void computeAccretionConditionImpl(size_t first, size_t last, Dataset& d, StarData& star)
{
    const double star_size2 = star.inner_size * star.inner_size;

    RemovalStatistics accreted_local{}, removed_local{};

    auto markForRemovalAndAdd = [&d](RemovalStatistics& statistics, size_t i)
    {
        d.keys[i]  = cstone::removeKey<typename Dataset::KeyType>::value;
        statistics = statistics + RemovalStatistics{d.m[i], {d.m[i] * d.vx[i], d.m[i] * d.vy[i], d.m[i] * d.vz[i]}, 1};
    };

#pragma omp declare reduction(add_statistics:RemovalStatistics : omp_out = omp_out + omp_in) initializer(omp_priv = {})

#pragma omp parallel for reduction(add_statistics : accreted_local, removed_local)
    for (size_t i = first; i < last; i++)
    {
        const double dx    = d.x[i] - star.position[0];
        const double dy    = d.y[i] - star.position[1];
        const double dz    = d.z[i] - star.position[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;

        const double r_cyl2 = d.x[i] * d.x[i] + d.y[i] * d.y[i];
        const double abs_z  = std::abs(d.z[i]);

        if (dist2 < star_size2) { markForRemovalAndAdd(accreted_local, i); }
        else if ((d.h[i] > star.removal_limit_h)
                || (r_cyl2 > star.removal_limit_r * star.removal_limit_r)
                || (abs_z  > star.removal_limit_z)) { markForRemovalAndAdd(removed_local, i); }
    }

    star.accreted_local = accreted_local;
    star.removed_local  = removed_local;
}

template<typename Dataset>
void computeBinaryAccretionConditionImpl(size_t first, size_t last, Dataset& d, StarData& star1, StarData& star2)
{
    const double star_size2_1 = star1.inner_size * star1.inner_size;
    const double star_size2_2 = star2.inner_size * star2.inner_size;

    RemovalStatistics accreted1_local{}, removed1_local{};
    RemovalStatistics accreted2_local{}, removed2_local{};

    auto markForRemovalAndAdd = [&d](RemovalStatistics& statistics, size_t i)
    {
        d.keys[i] = cstone::removeKey<typename Dataset::KeyType>::value;

        const double mi  = d.m[i];
        const double xi  = d.x[i],  yi  = d.y[i],  zi  = d.z[i];
        const double vxi = d.vx[i], vyi = d.vy[i], vzi = d.vz[i];

        // Positional aggregate init order: mass, momentum, count (unsigned),
        // weighted_position, angular_momentum.
        // '1u' is an unsigned literal matching the 'unsigned count' field.
        statistics = statistics + RemovalStatistics{
            mi,
            {mi * vxi, mi * vyi, mi * vzi},           // momentum = m*v
            1u,                                        // count
            {mi * xi,  mi * yi,  mi * zi},             // weighted_position = m*r
            {mi * (yi*vzi - zi*vyi),                   // angular_momentum = m*(r×v)
             mi * (zi*vxi - xi*vzi),
             mi * (xi*vyi - yi*vxi)}
        };
    };

#pragma omp declare reduction(add_statistics : RemovalStatistics : omp_out = omp_out + omp_in) initializer(omp_priv = {})

#pragma omp parallel for reduction(add_statistics : accreted1_local, removed1_local, accreted2_local, removed2_local)
    for (size_t i = first; i < last; i++)
    {
        const double dx1     = d.x[i] - star1.position[0];
        const double dy1     = d.y[i] - star1.position[1];
        const double dz1     = d.z[i] - star1.position[2];
        const double dist2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;

        const double dx2     = d.x[i] - star2.position[0];
        const double dy2     = d.y[i] - star2.position[1];
        const double dz2     = d.z[i] - star2.position[2];
        const double dist2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

        const double r_cyl2 = d.x[i] * d.x[i] + d.y[i] * d.y[i];
        const double abs_z  = std::abs(d.z[i]);

        if (dist2_1 < star_size2_1) { markForRemovalAndAdd(accreted1_local, i); }
        else if ((d.h[i] > star1.removal_limit_h)
                 || (r_cyl2 > star1.removal_limit_r * star1.removal_limit_r)
                 || (abs_z  > star1.removal_limit_z)) { markForRemovalAndAdd(removed1_local, i); }

        if (dist2_2 < star_size2_2) { markForRemovalAndAdd(accreted2_local, i); }
        else if ((d.h[i] > star2.removal_limit_h)
                 || (r_cyl2 > star2.removal_limit_r * star2.removal_limit_r)
                 || (abs_z  > star2.removal_limit_z)) { markForRemovalAndAdd(removed2_local, i); }
    }

    star1.accreted_local = accreted1_local;
    star1.removed_local  = removed1_local;
    star2.accreted_local = accreted2_local;
    star2.removed_local  = removed2_local;
}

} // namespace disk
