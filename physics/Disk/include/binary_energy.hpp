/*! @file
 * @brief Compute kinetic and potential energy for the two stars of a binary system
 *
 * @author Federico Ragno
 */
#pragma once

#include "star_data.hpp"

namespace disk
{

/*! @brief Compute and store ecin, egrav, etot for each star.
 *
 * Must be called after both stars' disk forces have been allreduced (so that star.potential is set)
 * and after computeBinaryForce (so that star.force_binary is set), but BEFORE star positions are
 * integrated, so that star.position_m1 still holds dx_n and gives velocity v_{n+1/2}.
 *
 * Energy accounting (each physical interaction counted exactly once):
 *   star.ecin  = 0.5 * M * |v|^2                      (star kinetic energy)
 *   star.egrav = M * star.potential                     (disk-star interaction PE)
 *              + 0.5 * star.force_binary[0]             (half of star-star PE, to avoid double-counting)
 *   star.etot  = star.ecin + star.egrav
 *
 * Note: star.potential is specific potential [L^2/T^2], not energy. Multiplying by M gives energy.
 * Note: star.force_binary[0] is the specific binary potential at the star = -G*M_other/r [L^2/T^2].
 *       M * (0.5 * force_binary[0]) = 0.5 * V12 counts the star-star PE exactly once.
 *
 * Total system energy (all contributions, no double-counting):
 *   E_total = d.etot + star1.etot + star2.etot
 * where d.etot = disk kinetic + disk internal + disk self-gravity (particle-particle, from ryoanji).
 */
template<typename Dataset, typename StarData>
void computeBinaryEnergy(StarData& star1, StarData& star2, const Dataset& d)
{

    auto computeStarEnergy = [&d](StarData& star)
    {
        double vx = star.position_m1[0] / d.minDt_m1;   // dx_n / dt_n = v_{n+1/2}
        double vy = star.position_m1[1] / d.minDt_m1;
        double vz = star.position_m1[2] / d.minDt_m1;
        star.ecin  = 0.5 * star.m * (vx*vx + vy*vy + vz*vz);
        star.egrav = star.m * (star.potential + 0.5 * star.force_binary[0]);
        star.etot  = star.ecin + star.egrav;
    };

    computeStarEnergy(star1);
    computeStarEnergy(star2);
}

} // namespace disk
