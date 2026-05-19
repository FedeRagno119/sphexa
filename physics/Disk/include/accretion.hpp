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
    const auto [m_acc1, p_acc1, m_rem1, p_rem1,
                m_acc2, p_acc2, m_rem2, p_rem2] = buffer::mpiAllreduceSum(
        star1.accreted_local.mass, star1.accreted_local.momentum,
        star1.removed_local.mass,  star1.removed_local.momentum,
        star2.accreted_local.mass, star2.accreted_local.momentum,
        star2.removed_local.mass,  star2.removed_local.momentum);

    const double m_star1_new = m_acc1 + star1.m;
    std::array<double, 3> p_star1;
    for (size_t i = 0; i < 3; i++)
    {
        p_star1[i]           = (star1.position_m1[i] / minDt_m1) * star1.m + p_acc1[i];
        star1.position_m1[i] = p_star1[i] / m_star1_new * minDt_m1;
    }
    star1.m = m_star1_new;

    const double m_star2_new = m_acc2 + star2.m;
    std::array<double, 3> p_star2;
    for (size_t i = 0; i < 3; i++)
    {
        p_star2[i]           = (star2.position_m1[i] / minDt_m1) * star2.m + p_acc2[i];
        star2.position_m1[i] = p_star2[i] / m_star2_new * minDt_m1;
    }
    star2.m = m_star2_new;

    if (rank == 0)
    {
        std::printf("Star1 - removed mass: %g\taccreted mass: %g\tstar mass: %g\n", m_rem1, m_acc1, star1.m);
        std::printf("Star1 - removed momentum x: %g\taccreted momentum x: %g\tstar momentum x: %g\n",
                    p_rem1[0], p_acc1[0], p_star1[0]);
        std::printf("Star1 - removed momentum y: %g\taccreted momentum y: %g\tstar momentum y: %g\n",
                    p_rem1[1], p_acc1[1], p_star1[1]);
        std::printf("Star1 - removed momentum z: %g\taccreted momentum z: %g\tstar momentum z: %g\n",
                    p_rem1[2], p_acc1[2], p_star1[2]);
        std::printf("Star1 - accreted mass local: %g\n", star1.accreted_local.mass);

        std::printf("Star2 - removed mass: %g\taccreted mass: %g\tstar mass: %g\n", m_rem2, m_acc2, star2.m);
        std::printf("Star2 - removed momentum x: %g\taccreted momentum x: %g\tstar momentum x: %g\n",
                    p_rem2[0], p_acc2[0], p_star2[0]);
        std::printf("Star2 - removed momentum y: %g\taccreted momentum y: %g\tstar momentum y: %g\n",
                    p_rem2[1], p_acc2[1], p_star2[1]);
        std::printf("Star2 - removed momentum z: %g\taccreted momentum z: %g\tstar momentum z: %g\n",
                    p_rem2[2], p_acc2[2], p_star2[2]);
        std::printf("Star2 - accreted mass local: %g\n", star2.accreted_local.mass);
    }
}

} // namespace disk
