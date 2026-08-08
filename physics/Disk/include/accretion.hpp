//
// Created by Noah Kubli on 15.03.2024.
//

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>

#include "accretion_impl.hpp"
#include "accretion_gpu.hpp"
#include "buffer_reduce.hpp"
#include "get_ptr.hpp"

namespace disk
{

//! @brief Flag particles for removal. Overwrites keys for the removed particles.
template<typename Dataset, typename StarData>
void computeAccretionCondition(size_t first, size_t last, Dataset& d, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeAccretionConditionGPU(first, last, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d), getPtr<"h">(d),
                                     getPtr<"keys">(d), getPtr<"m">(d), getPtr<"vx">(d), getPtr<"vy">(d),
                                     getPtr<"vz">(d), star);
    }
    else { computeAccretionConditionImpl(first, last, d, star); }
}

//! @brief Exchange accreted mass and momentum between ranks and add to star.
template<typename StarData>
void exchangeAndAccreteOnStar(StarData& star, double minDt_m1, int rank)
{
    const auto [m_accreted, p_accreted, m_removed, p_removed] = buffer::mpiAllreduceSum(
        star.accreted_local.mass, star.accreted_local.momentum, star.removed_local.mass, star.removed_local.momentum);

    const double m_star_new = m_accreted + star.m;

    std::array<double, 3> p_star;
    for (size_t i = 0; i < 3; i++)
    {
        p_star[i] = (star.position_m1[i] / minDt_m1) * star.m;
        p_star[i] += p_accreted[i];
        star.position_m1[i] = p_star[i] / m_star_new * minDt_m1;
    }

    star.m = m_star_new;
    if (rank == 0)
    {
        std::printf("removed mass: %g\taccreted mass: %g\tstar mass: %g\n", m_removed, m_accreted, star.m);
        std::printf("removed momentum x: %g\taccreted momentum x: %g\tstar momentum x: %g\n", p_removed[0],
                    p_accreted[0], p_star[0]);
        std::printf("removed momentum y: %g\taccreted momentum y: %g\tstar momentum y: %g\n", p_removed[1],
                    p_accreted[1], p_star[1]);
        std::printf("removed momentum z: %g\taccreted momentum z: %g\tstar momentum z: %g\n", p_removed[2],
                    p_accreted[2], p_star[2]);
        std::printf("accreted mass local: %g\n", star.accreted_local.mass);
    }
}

/*! @brief Tally mass/energy of particles about to be removed for failing neighbor-search convergence,
 *  strictly before sph::updateSmoothingLength() actually removes them. Must be called with the same
 *  [first,last) range and on the same nc[] snapshot that updateSmoothingLength() will subsequently use.
 */
template<typename Dataset, typename StarData>
MassEnergyTally tallyNonConvergedRemoval(size_t first, size_t last, Dataset& d, const StarData& star1,
                                         const StarData& star2)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        return tallyNonConvergedRemovalGPU(first, last, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d),
                                           getPtr<"vx">(d), getPtr<"vy">(d), getPtr<"vz">(d), getPtr<"m">(d),
                                           getPtr<"u">(d), getPtr<"ugrav">(d), getPtr<"nc">(d), d.g, star1.position,
                                           star1.m, star1.grav_softening, star2.position, star2.m,
                                           star2.grav_softening);
    }
    else { return tallyNonConvergedRemovalImpl(first, last, d, star1, star2); }
}

template<typename Dataset, typename StarData>
void computeBinaryAccretionCondition(size_t first, size_t last, Dataset& d, StarData& star1, StarData& star2)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeBinaryAccretionConditionGPU(first, last,
                                           getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d), getPtr<"h">(d),
                                           getPtr<"keys">(d), getPtr<"m">(d),
                                           getPtr<"vx">(d), getPtr<"vy">(d), getPtr<"vz">(d),
                                           getPtr<"u">(d), getPtr<"ugrav">(d), d.g,
                                           star1, star2);
    }
    else { computeBinaryAccretionConditionImpl(first, last, d, star1, star2); }
}

/*! @brief Reduce accretion/removal tallies across ranks and apply them to the two sinks.
 *
 * @return the globally-reduced {mass, energy} of particles removed by the limit_h/r/z cutoffs. These are
 *         star-independent (the cutoffs never reference a star position), so they are returned for the
 *         caller to accumulate globally rather than being stored per-star. Only star1.removed_local is
 *         populated upstream; star2.removed_local is always zero and is not read here.
 */
/*! @param minDt  the *current* step's timestep. This function runs after computeAndExchangeBinaryPositions,
 *                which rebuilt star.position_m1 as the displacement over the current step (dt = minDt).
 *                The star velocity is therefore position_m1/minDt (NOT position_m1/minDt_m1). Using
 *                minDt_m1 here left the reconstructed v_old off by minDt/minDt_m1, biasing the dynamics'
 *                accretion recoil and the accreted_ke/L diagnostics by the per-step timestep change
 *                (~few %). accreted_p is unaffected either way (the v_old term cancels: Δp = Σ mₐvₐ).
 */
template<typename StarData>
MassEnergyTally exchangeAndAccreteBinaryStars(StarData& star1, StarData& star2, double minDt, int rank)
{
    // Single allreduce: accreted (6 quantities × 2 stars) + the star-independent removal tally (3)
    const auto [m_acc1, p_acc1, wp_acc1, wL_acc1, count_acc1, energy_acc1, m_rem, p_rem, energy_rem,
                m_acc2, p_acc2, wp_acc2, wL_acc2, count_acc2, energy_acc2]
        = buffer::mpiAllreduceSum(
            star1.accreted_local.mass,
            star1.accreted_local.momentum,
            star1.accreted_local.weighted_position,
            star1.accreted_local.angular_momentum,
            star1.accreted_local.count,
            star1.accreted_local.energy,
            star1.removed_local.mass,
            star1.removed_local.momentum,
            star1.removed_local.energy,
            star2.accreted_local.mass,
            star2.accreted_local.momentum,
            star2.accreted_local.weighted_position,
            star2.accreted_local.angular_momentum,
            star2.accreted_local.count,
            star2.accreted_local.energy);

    auto accrete = [minDt](StarData& star, double m_acc,
                               const cstone::Vec3<double>& p_acc,
                               const cstone::Vec3<double>& wp_acc,
                               const cstone::Vec3<double>& wL_acc,
                               uint64_t count_acc, double energy_acc)
    {
        // Particle-side sums (count, pre-accretion energy) don't depend on m_acc being non-zero in
        // principle, but they can only be non-zero together with m_acc, so gating on it is equivalent.
        if (m_acc == 0.0) { return; }

        const double M_old    = star.m;
        const double M_new    = M_old + m_acc;
        const double inv_Mnew = 1.0 / M_new;

        // Save pre-accretion r and v BEFORE modifying position/position_m1
        // (required for the spin formula which uses the old COM orbital angular momentum)
        const cstone::Vec3<double> r_old = star.position;
        cstone::Vec3<double> v_old;
        for (int k = 0; k < 3; k++) v_old[k] = star.position_m1[k] / minDt;

        // Eq 151: r_new = (M_old*r_old + Σ m_a*r_a) / M_new
        for (int k = 0; k < 3; k++)
            star.position[k] = (M_old * r_old[k] + wp_acc[k]) * inv_Mnew;

        // Eq 152: v_new stored back as the leapfrog displacement over the current step = v_new * minDt
        cstone::Vec3<double> p_star;
        for (int k = 0; k < 3; k++)
            p_star[k] = M_old * v_old[k] + p_acc[k];
        for (int k = 0; k < 3; k++)
            star.position_m1[k] = p_star[k] * inv_Mnew * minDt;

        // Eq 154: S_new = S_old + M_old*(r_old×v_old) + wL_acc - M_new*(r_new×v_new)
        // wL_acc = Σ m_a*(r_a×v_a); r_new/v_new are the updated COM position and velocity
        cstone::Vec3<double> v_new;
        for (int k = 0; k < 3; k++) v_new[k] = p_star[k] * inv_Mnew;
        const cstone::Vec3<double> r_new = star.position;
        star.spin = star.spin
                    + M_old * util::cross(r_old, v_old)
                    + wL_acc
                    - M_new * util::cross(r_new, v_new);

        // Diagnostics: items 1/3/5/6 are all before/after deltas of the star's own bulk state
        // (M, r, v) -- not sums over the accreted particles' individual contributions.
        star.accreted_mass_accum += m_acc; // == M_new - M_old
        star.accreted_ke_accum   += 0.5 * M_new * util::dot(v_new, v_new) - 0.5 * M_old * util::dot(v_old, v_old);
        for (int k = 0; k < 3; k++) star.accreted_p_accum[k] += M_new * v_new[k] - M_old * v_old[k];
        const cstone::Vec3<double> L_old = M_old * util::cross(r_old, v_old);
        const cstone::Vec3<double> L_new = M_new * util::cross(r_new, v_new);
        for (int k = 0; k < 3; k++) star.accreted_L_accum[k] += L_new[k] - L_old[k];

        // Items 2/4: genuine particle-side sums, already reduced above.
        star.accreted_count_accum  += count_acc;
        star.accreted_energy_accum += energy_acc;

        // Eq 155
        star.m = M_new;
    };

    accrete(star1, m_acc1, p_acc1, wp_acc1, wL_acc1, count_acc1, energy_acc1);
    accrete(star2, m_acc2, p_acc2, wp_acc2, wL_acc2, count_acc2, energy_acc2);

    if (rank == 0)
    {
        std::printf("Star1 - accreted: %g\tmass: %g\n", m_acc1, star1.m);
        std::printf("Star1 - spin: %g %g %g\n", star1.spin[0], star1.spin[1], star1.spin[2]);
        std::printf("Star2 - accreted: %g\tmass: %g\n", m_acc2, star2.m);
        std::printf("Star2 - spin: %g %g %g\n", star2.spin[0], star2.spin[1], star2.spin[2]);
        std::printf("Limit-removed (global) - mass: %g\tenergy: %g\n", m_rem, energy_rem);
    }

    return {m_rem, energy_rem};
}

} // namespace disk
