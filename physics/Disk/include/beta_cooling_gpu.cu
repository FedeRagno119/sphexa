//
// Created by Noah Kubli on 17.04.2024.
//

#include "beta_cooling_gpu.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "star_data.hpp"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <cmath>

namespace disk
{

template<typename Treal, typename Thydro, typename Ts>
__global__ void betaCoolingGPUKernel(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                     const Treal* u, const Thydro* rho, Treal* du, Treal g, Ts star_mass,
                                     cstone::Vec3<Ts> star_position, Ts beta, Ts u_floor, Ts cooling_rho_limit,
                                     Ts beta_b, Ts r_beta0)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }
    if (rho[i] >= cooling_rho_limit || u[i] <= u_floor) return;

    const double dx         = x[i] - star_position[0];
    const double dy         = y[i] - star_position[1];
    const double dz         = z[i] - star_position[2];
    const double dist2      = dx * dx + dy * dy + dz * dz;
    const double dist       = sqrt(dist2);
    const double omega      = sqrt(g * star_mass / (dist2 * dist));
    const double beta_local = beta * pow(dist / r_beta0, -beta_b);
    du[i] += -u[i] * omega / beta_local;
}

template<typename Treal, typename Thydro>
void betaCoolingGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, const Treal* u,
                    const Thydro* rho, Treal* du, const Treal g, const StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    betaCoolingGPUKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, u, rho, du, g, star.m, star.position,
                                                    star.beta, star.u_floor, star.cooling_rho_limit,
                                                    star.beta_b, star.r_beta0);

    checkGpuErrors(cudaDeviceSynchronize());
}

#define BETA_COOLING_GPU(Treal, Thydro)                                                                                \
    template void betaCoolingGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,            \
                                 const Treal* u, const Thydro* rho, Treal* du, const Treal g, const StarData& star);

BETA_COOLING_GPU(double, double);
BETA_COOLING_GPU(double, float);

template<typename Treal, typename Thydro, typename Ts>
__global__ void betaCoolingBinaryGPUKernel(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                            const Treal* u, const Thydro* rho, Treal* du, Treal* du_cool_accum,
                                            double minDt, Treal g,
                                            Ts M_total, cstone::Vec3<Ts> com_pos, Ts beta, Ts u_floor,
                                            Ts cooling_rho_limit, Ts betaEps, Ts beta_b, Ts r_beta0)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }
    if (rho[i] >= cooling_rho_limit || u[i] <= u_floor) return;

    const double dx         = x[i] - com_pos[0];
    const double dy         = y[i] - com_pos[1];
    const double dz         = z[i] - com_pos[2];
    const double dist2      = dx * dx + dy * dy + dz * dz + betaEps * betaEps;
    const double dist       = sqrt(dist2);
    const double omega      = sqrt(g * M_total / (dist2 * dist));
    const double beta_local = beta * pow(dist / r_beta0, -beta_b);
    const double du_cool    = -u[i] * omega / beta_local;
    // Nullable outputs: du is written in computeForces (for the timestep limiter); du_cool_accum is
    // written by a second call from integrate() after computeTimestep, where minDt is the current dt.
    if (du) { du[i] += du_cool; }
    if (du_cool_accum) { du_cool_accum[i] += du_cool * minDt; }
}

template<typename Treal, typename Thydro>
void betaCoolingBinaryGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, const Treal* u,
                           const Thydro* rho, Treal* du, Treal* du_cool_accum, double minDt, const Treal g,
                           const StarData& star1, const StarData& star2, bool applyDu, bool accumulateLoss)
{
    // Cooling center blends between the primary and the binary COM as the secondary approaches
    // (see betaCoolingCenter); M_eff blends correspondingly between m1 and the total mass.
    cstone::Vec3<double> center;
    double               M_eff;
    betaCoolingCenter(star1, star2, center, M_eff);

    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    betaCoolingBinaryGPUKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, u, rho,
                                                           applyDu ? du : nullptr,
                                                           accumulateLoss ? du_cool_accum : nullptr, minDt, g,
                                                           M_eff, center,
                                                           star1.beta, star1.u_floor, star1.cooling_rho_limit,
                                                           star1.betaEps, star1.beta_b, star1.r_beta0);
    checkGpuErrors(cudaDeviceSynchronize());
}

#define BETA_COOLING_BINARY_GPU(Treal, Thydro)                                                                         \
    template void betaCoolingBinaryGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,     \
                                        const Treal* u, const Thydro* rho, Treal* du, Treal* du_cool_accum,           \
                                        double minDt, const Treal g,                                                  \
                                        const StarData& star1, const StarData& star2, bool applyDu,                   \
                                        bool accumulateLoss);

BETA_COOLING_BINARY_GPU(double, double);
BETA_COOLING_BINARY_GPU(double, float);

template<typename Tu, typename Tdu>
struct AbsDivide
{
    double u_inf;
    HOST_DEVICE_FUN double operator()(const thrust::tuple<Tu, Tdu>& X) const
    {
        double du_val = double{thrust::get<1>(X)};
        if (du_val == 0.0) { return INFINITY; }
        // Use u_inf as a floor so a near-zero u still yields a finite, conservative dt.
        double u_eff = fmax(double{thrust::get<0>(X)}, u_inf);
        return fabs(u_eff / du_val);
    }
};

template<typename Treal>
double duTimestepGPU(size_t first, size_t last, const Treal* u, const Treal* du, double u_inf)
{
    using Tu  = std::decay_t<decltype(*u)>;
    using Tdu = std::decay_t<decltype(*du)>;

    auto begin = thrust::make_zip_iterator(u + first, du + first);
    auto end   = thrust::make_zip_iterator(u + last, du + last);

    double init = INFINITY;

    return thrust::transform_reduce(thrust::device, begin, end, AbsDivide<Tu, Tdu>{u_inf}, init,
                                    thrust::minimum<double>{});
}

#define DU_TIMESTEP_GPU(Treal)                                                                                         \
    template double duTimestepGPU(size_t first, size_t last, const Treal* u, const Treal* du, double u_inf);

DU_TIMESTEP_GPU(double);

// --- Percentile-based energy timestep ---

//! @brief Computes |max(u, u_inf) / du|; returns INFINITY for particles with du == 0.
template<typename Tu, typename Tdu>
struct ComputeRatio
{
    double u_inf;
    HOST_DEVICE_FUN double operator()(const thrust::tuple<Tu, Tdu>& X) const
    {
        double du_val = double{thrust::get<1>(X)};
        if (du_val == 0.0) { return INFINITY; }
        double u_eff = fmax(double{thrust::get<0>(X)}, u_inf);
        return fabs(u_eff / du_val);
    }
};

//! @brief Predicate: true for entries that represent invalid particles (du == 0).
struct NotFinite
{
    HOST_DEVICE_FUN bool operator()(double x) const { return !isfinite(x); }
};

//! @brief GPU percentile-based energy timestep: returns the k-th smallest ratio |u_eff/du|
//! where k = floor((1 - duLimitPercentile) * N_valid). When duLimitPercentile == 1.0,
//! k == 0 and the result equals the global minimum (same as duTimestepGPU).
template<typename Treal>
double duTimestepPercentileGPU(size_t first, size_t last, const Treal* u, const Treal* du,
                               double u_inf, double duLimitPercentile)
{
    using Tu  = std::decay_t<decltype(*u)>;
    using Tdu = std::decay_t<decltype(*du)>;

    const size_t N = last - first;
    // Step 1: compute all ratios into a temporary device vector.
    // Particles with du == 0 produce INFINITY and are filtered out in step 2.
    thrust::device_vector<double> ratios(N);
    auto zip_begin = thrust::make_zip_iterator(u + first, du + first);
    auto zip_end   = thrust::make_zip_iterator(u + last,  du + last);
    thrust::transform(thrust::device, zip_begin, zip_end,
                      ratios.begin(), ComputeRatio<Tu, Tdu>{u_inf});

    // Step 2: compact — move all invalid (non-finite) entries to the back, return iterator
    // to the new end of the valid prefix.
    auto valid_end = thrust::remove_if(thrust::device,
                                       ratios.begin(), ratios.end(), NotFinite{});
    const size_t N_valid = static_cast<size_t>(valid_end - ratios.begin());
    if (N_valid == 0) { return INFINITY; }

    // Step 3: sort the valid range and pick the k-th smallest element.
    // k=0 when duLimitPercentile==1.0, recovering the global minimum.
    const size_t k = std::min(size_t(std::floor((1.0 - duLimitPercentile) * N_valid)),
                              N_valid - 1);
    thrust::sort(thrust::device, ratios.begin(), valid_end);

    return double(ratios[k]);
}

#define DU_TIMESTEP_PERCENTILE_GPU(Treal)                                                      \
    template double duTimestepPercentileGPU(size_t first, size_t last,                        \
                                            const Treal* u, const Treal* du,                  \
                                            double u_inf, double duLimitPercentile);

DU_TIMESTEP_PERCENTILE_GPU(double);

// Persistent device counter — same pattern as force_device in central_force_gpu.cu.
static __device__ unsigned long long count_device;

template<typename Treal>
__global__ void applyEnergyFloorKernel(size_t first, size_t last, Treal* u, Treal u_min,
                                       unsigned long long* count)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }
    if (u[i] < u_min)
    {
        u[i] = u_min;
        atomicAdd(count, 1ULL);
    }
}

template<typename Treal>
size_t applyEnergyFloorGPU(size_t first, size_t last, Treal* u, double u_inf)
{
    const Treal u_min = Treal(u_inf) / Treal(10);

    const unsigned long long zero = 0;
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(count_device), &zero, sizeof(zero)));

    unsigned long long* count_ptr;
    checkGpuErrors(cudaGetSymbolAddress((void**)&count_ptr, GPU_SYMBOL(count_device)));

    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    applyEnergyFloorKernel<<<numBlocks, numThreads>>>(first, last, u, u_min, count_ptr);
    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());

    unsigned long long h_count;
    checkGpuErrors(cudaMemcpyFromSymbol(&h_count, GPU_SYMBOL(count_device), sizeof(h_count)));

    return size_t(h_count);
}

#define APPLY_ENERGY_FLOOR_GPU(Treal)                                                                                  \
    template size_t applyEnergyFloorGPU(size_t first, size_t last, Treal* u, double u_inf);

APPLY_ENERGY_FLOOR_GPU(double);

} // namespace disk
