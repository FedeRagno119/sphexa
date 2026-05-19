//
// Created by Noah Kubli on 17.04.2024.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "beta_cooling_gpu.hpp"
#include "get_ptr.hpp"

namespace disk
{

template<typename Dataset, typename StarData>
void betaCoolingImpl(size_t first, size_t last, Dataset& d, const StarData& star)
{
#pragma omp parallel for
    for (size_t i = first; i < last; i++)
    {
        if (d.rho[i] < star.cooling_rho_limit && d.u[i] > star.u_floor)
        {
            const double dx    = d.x[i] - star.position[0];
            const double dy    = d.y[i] - star.position[1];
            const double dz    = d.z[i] - star.position[2];
            const double dist2 = dx * dx + dy * dy + dz * dz;
            const double dist  = std::sqrt(dist2);
            const double omega = std::sqrt(d.g * star.m / (dist2 * dist));
            d.du[i] += -d.u[i] * omega / star.beta;
        }
    }
}

template<typename Dataset>
auto duTimestepImpl(size_t first, size_t last, const Dataset& d)
{
    using Tu         = std::decay_t<decltype(d.u[0])>;
    using Tdu        = std::decay_t<decltype(d.du[0])>;
    using Tt         = std::common_type_t<Tu, Tdu>;
    Tt duTimestepMin = std::numeric_limits<Tt>::infinity();
    const Tt u_inf   = Tt(d.u_inf);

#pragma omp parallel for reduction(min : duTimestepMin)
    for (size_t i = first; i < last; i++)
    {
        if (d.du[i] == 0) { continue; }
        // Use u_inf as a floor so a near-zero u still yields a finite, conservative dt.
        const Tt u_eff    = std::max(Tt(d.u[i]), u_inf);
        const Tt duTimestep = std::abs(u_eff / d.du[i]);
        duTimestepMin = std::min(duTimestepMin, duTimestep);
    }
    return duTimestepMin;
}

//! @brief Compute a maximal time step using the k-th order statistic of |u_eff/du|, where
//! k = floor((1 - d.duLimitPercentile) * N_valid). This excludes the (1-duLimitPercentile)
//! fraction of most-constraining particles from setting dt, preventing outliers from
//! stalling the simulation. When duLimitPercentile == 1.0 the result equals the minimum.
template<typename Dataset>
auto duTimestepPercentileImpl(size_t first, size_t last, const Dataset& d)
{
    using Tu  = std::decay_t<decltype(d.u[0])>;
    using Tdu = std::decay_t<decltype(d.du[0])>;
    using Tt  = std::common_type_t<Tu, Tdu>;

    const Tt u_inf = Tt(d.u_inf);

    // Step 1: collect valid ratios in parallel using thread-local vectors to avoid
    // false sharing and dynamic memory contention, then merge serially.
    std::vector<std::vector<Tt>> thread_ratios;
#pragma omp parallel
    {
        std::vector<Tt> local;
#pragma omp for nowait schedule(static)
        for (size_t i = first; i < last; i++)
        {
            if (d.du[i] == 0) { continue; }
            const Tt u_eff = std::max(Tt(d.u[i]), u_inf);
            local.push_back(std::abs(u_eff / Tt(d.du[i])));
        }
#pragma omp critical
        thread_ratios.push_back(std::move(local));
    }

    std::vector<Tt> ratios;
    for (auto& v : thread_ratios) { ratios.insert(ratios.end(), v.begin(), v.end()); }
    if (ratios.empty()) { return std::numeric_limits<Tt>::infinity(); }

    // Step 2: find the k-th smallest ratio in O(N) average via nth_element.
    // k=0 when duLimitPercentile==1.0, giving the global minimum (identical to duTimestepImpl).
    const size_t N = ratios.size();
    const size_t k = std::min(size_t(std::floor((1.0 - double(d.duLimitPercentile)) * N)), N - 1);
    std::nth_element(ratios.begin(), ratios.begin() + k, ratios.end());
    return ratios[k];
}

//! @brief CPU/GPU dispatcher for the percentile-based energy timestep.
template<typename Dataset, typename StarData>
void duTimestepPercentile(size_t first, size_t last, Dataset& d, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        star.t_du = star.K_u *
                    duTimestepPercentileGPU(first, last, getPtr<"u">(d), getPtr<"du">(d),
                                           d.u_inf, d.duLimitPercentile);
    }
    else { star.t_du = star.K_u * duTimestepPercentileImpl(first, last, d); }
}

//! @brief Clamp particle energies to u_inf/10 from below; returns the local count of clamped particles.
template<typename Dataset>
size_t applyEnergyFloorImpl(size_t first, size_t last, Dataset& d)
{
    using Tu       = std::decay_t<decltype(d.u[0])>;
    const Tu u_min = Tu(d.u_inf) / Tu(10);
    size_t   count = 0;

#pragma omp parallel for reduction(+ : count)
    for (size_t i = first; i < last; i++)
    {
        if (d.u[i] < u_min)
        {
            d.u[i] = u_min;
            ++count;
        }
    }
    return count;
}

//! @brief Clamp particle energies to u_inf/10; returns local count of clamped particles (0 if u_inf == 0).
template<typename Dataset>
size_t applyEnergyFloor(size_t first, size_t last, Dataset& d)
{
    if (d.u_inf == 0) { return 0; }

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        return applyEnergyFloorGPU(first, last, getPtr<"u">(d), d.u_inf);
    }
    else { return applyEnergyFloorImpl(first, last, d); }
}

//! @brief Cool the disk with cooling time proportional to the Keplerian orbital time around the binary center of mass
template<typename Dataset, typename StarData>
void betaCoolingBinaryImpl(size_t first, size_t last, Dataset& d, const StarData& star1, const StarData& star2)
{
    const double M_total = star1.m + star2.m;
    const double com_x   = (star1.m * star1.position[0] + star2.m * star2.position[0]) / M_total;
    const double com_y   = (star1.m * star1.position[1] + star2.m * star2.position[1]) / M_total;
    const double com_z   = (star1.m * star1.position[2] + star2.m * star2.position[2]) / M_total;
    const double eps     = star1.betaEps; //FR: smoothing length - same for both stars

#pragma omp parallel for
    for (size_t i = first; i < last; i++)
    {
        if (d.rho[i] < star1.cooling_rho_limit && d.u[i] > star1.u_floor)
        {
            const double dx    = d.x[i] - com_x;
            const double dy    = d.y[i] - com_y;
            const double dz    = d.z[i] - com_z;
            const double dist2 = dx * dx + dy * dy + dz * dz + eps * eps;
            const double dist  = std::sqrt(dist2);
            const double omega = std::sqrt(d.g * M_total / (dist2 * dist));
            d.du[i] += -d.u[i] * omega / star1.beta;
        }
    }
}

//! @brief Cool the disk with a cooling time proportional to the Keplerian orbital time (around the star)
template<typename Dataset, typename StarData>
void betaCooling(size_t startIndex, size_t endIndex, Dataset& d, const StarData& star)
{
    using T_beta = std::decay_t<decltype(star.beta)>;
    if (star.beta != std::numeric_limits<T_beta>::infinity())
    {
        if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
        {
            betaCoolingGPU(startIndex, endIndex, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d), getPtr<"u">(d),
                           getPtr<"rho">(d), getPtr<"du">(d), d.g, star);
        }
        else { betaCoolingImpl(startIndex, endIndex, d, star); }
    }
}

//! @brief Cool the disk with a cooling time proportional to the Keplerian orbital time around the binary center of mass
template<typename Dataset, typename StarData>
void betaCoolingBinary(size_t startIndex, size_t endIndex, Dataset& d, const StarData& star1, const StarData& star2)
{
    using T_beta = std::decay_t<decltype(star1.beta)>;
    if (star1.beta != std::numeric_limits<T_beta>::infinity())
    {
        if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
        {
            betaCoolingBinaryGPU(startIndex, endIndex, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d), getPtr<"u">(d),
                                 getPtr<"rho">(d), getPtr<"du">(d), d.g, star1, star2);
        }
        else { betaCoolingBinaryImpl(startIndex, endIndex, d, star1, star2); }
    }
}

//! @brief Compute a maximal time step depending on how fast the internal energy changes.
//! Dispatches to the percentile-based variant when d.duLimitPercentile < 1.0, otherwise
//! uses the global minimum.
template<typename Dataset, typename StarData>
void duTimestep(size_t startIndex, size_t endIndex, Dataset& d, StarData& star)
{
    if (star.K_u == std::numeric_limits<decltype(star.K_u)>::infinity())
    {
        star.t_du = std::numeric_limits<decltype(star.t_du)>::infinity();
        return;
    }
    if (d.duLimitPercentile < 1.0)
    {
        duTimestepPercentile(startIndex, endIndex, d, star);
        return;
    }
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        star.t_du = star.K_u * duTimestepGPU(startIndex, endIndex, getPtr<"u">(d), getPtr<"du">(d), d.u_inf);
    }
    else { star.t_du = star.K_u * duTimestepImpl(startIndex, endIndex, d); }
}

} // namespace disk
