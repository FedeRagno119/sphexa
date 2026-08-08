//
// Created by Noah Kubli on 14.03.2024.
//

#pragma once

#include "accretion_gpu.hpp"
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

    RemovalStatistics accreted1_local{}, accreted2_local{};
    RemovalStatistics removed_local{};   // star-independent: h/r/z limits don't reference a star position

    const double G = d.g;

    // Per-particle total energy (KE + internal + potential w.r.t. disk self-gravity + both stars),
    // evaluated just before the particle is removed/accreted. dist1/dist2 are the already-computed
    // particle-star separations from the calling loop.
    auto markForRemovalAndAdd = [&d, &star1, &star2, G](RemovalStatistics& statistics, size_t i, double dist1, double dist2)
    {
        d.keys[i] = cstone::removeKey<typename Dataset::KeyType>::value;

        const double mi  = d.m[i];
        const double xi  = d.x[i],  yi  = d.y[i],  zi  = d.z[i];
        const double vxi = d.vx[i], vyi = d.vy[i], vzi = d.vz[i];

        const double ke_i    = 0.5 * mi * (vxi * vxi + vyi * vyi + vzi * vzi);
        const double u_i     = mi * d.u[i];
        //FR: d.ugrav[i] is ALREADY a potential energy (m_i * Phi_i), not a specific potential.
        //    Multiplying by mi here double-weighted it by mass and effectively zeroed the term.
        const double ugrav_i = d.ugrav[i];
        const double star_pe_i = softenedStarPotentialEnergy(mi, G, star1.m, star1.grav_softening, dist1,
                                                             star2.m, star2.grav_softening, dist2);
        const double energy_i  = ke_i + u_i + ugrav_i + star_pe_i;

        // Positional aggregate init order: mass, momentum, count (unsigned),
        // weighted_position, angular_momentum, energy.
        // '1u' is an unsigned literal matching the 'unsigned count' field.
        statistics = statistics + RemovalStatistics{
            mi,
            {mi * vxi, mi * vyi, mi * vzi},           // momentum = m*v
            1u,                                        // count
            {mi * xi,  mi * yi,  mi * zi},             // weighted_position = m*r
            {mi * (yi*vzi - zi*vyi),                   // angular_momentum = m*(r×v)
             mi * (zi*vxi - xi*vzi),
             mi * (xi*vyi - yi*vxi)},
            energy_i
        };
    };

#pragma omp declare reduction(add_statistics : RemovalStatistics : omp_out = omp_out + omp_in) initializer(omp_priv = {})

#pragma omp parallel for reduction(add_statistics : accreted1_local, accreted2_local, removed_local)
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

        const double dist_1 = std::sqrt(dist2_1);
        const double dist_2 = std::sqrt(dist2_2);

        const double r_cyl2 = d.x[i] * d.x[i] + d.y[i] * d.y[i];
        const double abs_z  = std::abs(d.z[i]);

        /*! Branch order matters, and the chain is deliberately mutually exclusive.
         *
         *  Accretion is per-star (proximity to that sink), but the removal limits are *not*: h,
         *  cylindrical radius and |z| are properties of the particle relative to the origin and never
         *  reference a star position. star1 and star2 carry their own copies of removal_limit_h/r/z, but
         *  they describe the same global cutoff.
         *
         *  Previously this was two independent if/else-if blocks, so a particle failing the limits was
         *  passed to markForRemovalAndAdd twice -- once per star -- double-counting its mass and energy
         *  and making star1::removed_*_accum and star2::removed_*_accum bit-identical in every dump.
         *  A single chain tallies each removed particle exactly once, into a star-independent bucket.
         *
         *  Making accretion mutually exclusive also closes a latent double-count: a particle inside both
         *  sinks' inner_size spheres used to be added to both stars, creating mass from nothing. That
         *  needs the sinks to be within inner_size1+inner_size2 of each other -- impossible at the current
         *  separation, but it is a real hazard as the binary hardens toward merger.
         */
        const bool exceedsRemovalLimits = (d.h[i] > star1.removal_limit_h)
                                          || (r_cyl2 > star1.removal_limit_r * star1.removal_limit_r)
                                          || (abs_z > star1.removal_limit_z);

        if (dist2_1 < star_size2_1) { markForRemovalAndAdd(accreted1_local, i, dist_1, dist_2); }
        else if (dist2_2 < star_size2_2) { markForRemovalAndAdd(accreted2_local, i, dist_1, dist_2); }
        else if (exceedsRemovalLimits) { markForRemovalAndAdd(removed_local, i, dist_1, dist_2); }
    }

    star1.accreted_local = accreted1_local;
    star2.accreted_local = accreted2_local;

    // Star-independent by construction (see above): reported once, on star1, with star2 zeroed so the
    // two are never summed into a double count downstream.
    star1.removed_local = removed_local;
    star2.removed_local = RemovalStatistics{};
}

/*! @brief Tally mass and total energy of particles about to be removed for failing neighbor-search
 *  convergence (nc[i] <= 1), *before* sph::updateSmoothingLength() actually removes them.
 *
 *  This mirrors (does not replace) the nc[i] <= 1 condition in sph::updateSmoothingLengthCpu -- it must
 *  run strictly before that call, on the same nc[] snapshot, so the tally reflects exactly the particles
 *  that are about to disappear. Not attributed to either star (this failure isn't tied to proximity to
 *  a sink), so the result is a global (not per-star) mass/energy pair.
 */
template<typename Dataset>
MassEnergyTally tallyNonConvergedRemovalImpl(size_t first, size_t last, Dataset& d, const StarData& star1,
                                             const StarData& star2)
{
    const double G = d.g;

    double mass_sum{0.};
    double energy_sum{0.};

#pragma omp parallel for reduction(+ : mass_sum, energy_sum)
    for (size_t i = first; i < last; i++)
    {
        if (d.nc[i] > 1) { continue; }

        const double mi  = d.m[i];
        const double vxi = d.vx[i], vyi = d.vy[i], vzi = d.vz[i];

        const double dx1   = d.x[i] - star1.position[0];
        const double dy1   = d.y[i] - star1.position[1];
        const double dz1   = d.z[i] - star1.position[2];
        const double dist1 = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        const double dx2   = d.x[i] - star2.position[0];
        const double dy2   = d.y[i] - star2.position[1];
        const double dz2   = d.z[i] - star2.position[2];
        const double dist2 = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

        const double ke_i      = 0.5 * mi * (vxi * vxi + vyi * vyi + vzi * vzi);
        const double u_i       = mi * d.u[i];
        //FR: d.ugrav[i] is ALREADY a potential energy (m_i * Phi_i) -- do not multiply by mi.
        const double ugrav_i   = d.ugrav[i];
        const double star_pe_i = softenedStarPotentialEnergy(mi, G, star1.m, star1.grav_softening, dist1,
                                                             star2.m, star2.grav_softening, dist2);

        mass_sum += mi;
        energy_sum += ke_i + u_i + ugrav_i + star_pe_i;
    }

    return {mass_sum, energy_sum};
}

} // namespace disk
