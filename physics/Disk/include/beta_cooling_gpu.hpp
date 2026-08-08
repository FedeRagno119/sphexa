//
// Created by Noah Kubli on 17.04.2024.
//

#pragma once

#include "star_data.hpp"

namespace disk
{

template<typename Treal, typename Thydro>
void betaCoolingGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, const Treal* u,
                    const Thydro* rho, Treal* du, const Treal g, const StarData& star);

template<typename Treal, typename Thydro>
void betaCoolingBinaryGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, const Treal* u,
                           const Thydro* rho, Treal* du, Treal* du_cool_accum, double minDt, const Treal g,
                           const StarData& star1, const StarData& star2, bool applyDu, bool accumulateLoss);

template<typename Treal>
double duTimestepGPU(size_t first, size_t last, const Treal* u, const Treal* du, double u_inf);

template<typename Treal>
double duTimestepPercentileGPU(size_t first, size_t last, const Treal* u, const Treal* du,
                               double u_inf, double duLimitPercentile);

template<typename Treal>
size_t applyEnergyFloorGPU(size_t first, size_t last, Treal* u, double u_inf);

} // namespace disk
