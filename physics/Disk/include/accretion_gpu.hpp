//
// Created by Noah Kubli on 12.03.2024.
//

#pragma once

#include <cmath>

#include "star_data.hpp"

namespace disk
{

/*! @brief Potential energy of a particle of mass @p mi in the two stars' gravitational fields, using the
 *         SAME Plummer softening the dynamics uses (central_potential.hpp): −G·m_star/√(r² + soft²).
 *
 *  @param dist1,dist2  the UNSOFTENED particle–star separations; @param soft1,soft2 = each star's
 *         grav_softening. Shared by the CPU and GPU accretion/removal energy tallies so they stay in
 *         step with each other and with the force calculation. An unsoftened −G·m/r here (the previous
 *         code) overestimates the binding energy of particles inside the softening length — precisely the
 *         accreted particles, since inner_size ≈ grav_softening — and diverges as r→0.
 */
HOST_DEVICE_FUN inline double softenedStarPotentialEnergy(double mi, double G, double m1, double soft1,
                                                          double dist1, double m2, double soft2, double dist2)
{
    const double d1 = std::sqrt(dist1 * dist1 + soft1 * soft1);
    const double d2 = std::sqrt(dist2 * dist2 + soft2 * soft2);
    return mi * (-G * m1 / d1 - G * m2 / d2);
}

/*! @brief Plain mass/energy pair for the non-convergence removal tally.
 *
 *  Deliberately not std::pair<double,double>: returning std::pair by value triggers a GCC "note" about
 *  a C++14/C++17 ABI parameter-passing change for libstdc++ types with explicitly-defaulted special
 *  member functions. A plain aggregate with only implicit special members avoids it.
 */
struct MassEnergyTally
{
    double mass{0.};
    double energy{0.};

    HOST_DEVICE_FUN friend MassEnergyTally operator+(const MassEnergyTally& a, const MassEnergyTally& b)
    {
        return {a.mass + b.mass, a.energy + b.energy};
    }
};

template<typename Treal, typename Thydro, typename Tmass, typename Tnc>
MassEnergyTally tallyNonConvergedRemovalGPU(size_t first, size_t last, const Treal* x, const Treal* y,
                                            const Treal* z, const Thydro* vx, const Thydro* vy,
                                            const Thydro* vz, const Tmass* m, const Treal* u,
                                            const Thydro* ugrav, const Tnc* nc, double G,
                                            cstone::Vec3<double> star1_position, double star1_mass,
                                            double star1_grav_softening,
                                            cstone::Vec3<double> star2_position, double star2_mass,
                                            double star2_grav_softening);
template<typename Treal, typename Thydro, typename Tkeys, typename Tmass>
void computeAccretionConditionGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                  const Thydro* h, Tkeys* keys, const Tmass* m, const Thydro* vx, const Thydro* vy,
                                  const Thydro* vz, StarData& star);

template<typename Treal, typename Thydro, typename Tkeys, typename Tmass>
void computeBinaryAccretionConditionGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                         const Thydro* h, Tkeys* keys, const Tmass* m, const Thydro* vx,
                                         const Thydro* vy, const Thydro* vz, const Treal* u, const Thydro* ugrav,
                                         double G, StarData& star1, StarData& star2);
}
