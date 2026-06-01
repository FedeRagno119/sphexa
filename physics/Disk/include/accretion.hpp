//
// Created by Noah Kubli on 15.03.2024.
//

#pragma once

#include <array>
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

template<typename Dataset, typename StarData>
void computeBinaryAccretionCondition(size_t first, size_t last, Dataset& d, StarData& star1, StarData& star2)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeBinaryAccretionConditionGPU(first, last,
                                           getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d), getPtr<"h">(d),
                                           getPtr<"keys">(d), getPtr<"m">(d),
                                           getPtr<"vx">(d), getPtr<"vy">(d), getPtr<"vz">(d),
                                           star1, star2);
    }
    else { computeBinaryAccretionConditionImpl(first, last, d, star1, star2); }
}

template<typename StarData>
void exchangeAndAccreteBinaryStars(StarData& star1, StarData& star2, double minDt_m1, int rank)
{
    // Single allreduce: accreted (4 quantities × 2 stars) + removed diagnostics (2 × 2 stars)
    const auto [m_acc1, p_acc1, wp_acc1, wL_acc1, m_rem1, p_rem1,
                m_acc2, p_acc2, wp_acc2, wL_acc2, m_rem2, p_rem2]
        = buffer::mpiAllreduceSum(
            star1.accreted_local.mass,
            star1.accreted_local.momentum,
            star1.accreted_local.weighted_position,
            star1.accreted_local.angular_momentum,
            star1.removed_local.mass,
            star1.removed_local.momentum,
            star2.accreted_local.mass,
            star2.accreted_local.momentum,
            star2.accreted_local.weighted_position,
            star2.accreted_local.angular_momentum,
            star2.removed_local.mass,
            star2.removed_local.momentum);

    auto accrete = [minDt_m1](StarData& star, double m_acc,
                               const cstone::Vec3<double>& p_acc,
                               const cstone::Vec3<double>& wp_acc,
                               const cstone::Vec3<double>& wL_acc)
    {
        if (m_acc == 0.0) { return; }

        const double M_old    = star.m;
        const double M_new    = M_old + m_acc;
        const double inv_Mnew = 1.0 / M_new;

        // Save pre-accretion r and v BEFORE modifying position/position_m1
        // (required for the spin formula which uses the old COM orbital angular momentum)
        const cstone::Vec3<double> r_old = star.position;
        cstone::Vec3<double> v_old;
        for (int k = 0; k < 3; k++) v_old[k] = star.position_m1[k] / minDt_m1;

        // Eq 151: r_new = (M_old*r_old + Σ m_a*r_a) / M_new
        for (int k = 0; k < 3; k++)
            star.position[k] = (M_old * r_old[k] + wp_acc[k]) * inv_Mnew;

        // Eq 152: v_new stored as leapfrog displacement = v_new * minDt_m1
        cstone::Vec3<double> p_star;
        for (int k = 0; k < 3; k++)
            p_star[k] = M_old * v_old[k] + p_acc[k];
        for (int k = 0; k < 3; k++)
            star.position_m1[k] = p_star[k] * inv_Mnew * minDt_m1;

        // Eq 154: S_new = S_old + M_old*(r_old×v_old) + wL_acc - M_new*(r_new×v_new)
        // wL_acc = Σ m_a*(r_a×v_a); r_new/v_new are the updated COM position and velocity
        cstone::Vec3<double> v_new;
        for (int k = 0; k < 3; k++) v_new[k] = p_star[k] * inv_Mnew;
        const cstone::Vec3<double> r_new = star.position;
        star.spin = star.spin
                    + M_old * util::cross(r_old, v_old)
                    + wL_acc
                    - M_new * util::cross(r_new, v_new);

        // Eq 155
        star.m = M_new;
    };

    accrete(star1, m_acc1, p_acc1, wp_acc1, wL_acc1);
    accrete(star2, m_acc2, p_acc2, wp_acc2, wL_acc2);

    if (rank == 0)
    {
        std::printf("Star1 - removed: %g\taccreted: %g\tmass: %g\n", m_rem1, m_acc1, star1.m);
        std::printf("Star1 - spin: %g %g %g\n", star1.spin[0], star1.spin[1], star1.spin[2]);
        std::printf("Star2 - removed: %g\taccreted: %g\tmass: %g\n", m_rem2, m_acc2, star2.m);
        std::printf("Star2 - spin: %g %g %g\n", star2.spin[0], star2.spin[1], star2.spin[2]);
    }
}

} // namespace disk
