//
// Created by Noah Kubli on 12.03.2024.
//

#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/tree/definitions.h"

#include "accretion_gpu.hpp"
#include "star_data.hpp"
#include "removal_statistics.hpp"

namespace disk
{

__device__ void atomicAddRS(RemovalStatistics* x, const RemovalStatistics& y)
{
    atomicAdd(&(x->mass), y.mass);
    atomicAdd(&(x->momentum[0]), y.momentum[0]);
    atomicAdd(&(x->momentum[1]), y.momentum[1]);
    atomicAdd(&(x->momentum[2]), y.momentum[2]);
    atomicAdd(&(x->count), y.count);
}

template<typename Tkeys, typename Tm, typename Tv>
__device__ void markForRemovalAndAdd(RemovalStatistics& statistics, size_t i, Tkeys* keys, const Tm* m, const Tv* vx,
                                     const Tv* vy, const Tv* vz)
{
    keys[i]                = cstone::removeKey<Tkeys>::value;
    statistics.mass        = m[i];
    statistics.momentum[0] = m[i] * vx[i];
    statistics.momentum[1] = m[i] * vy[i];
    statistics.momentum[2] = m[i] * vz[i];
    statistics.count       = 1;
}

template<unsigned numThreads, typename T1, typename Th, typename Tkeys, typename T2, typename Tm, typename Tv>
__global__ void computeAccretionConditionKernel(size_t first, size_t last, const T1* x, const T1* y, const T1* z,
                                                const Th* h, Tkeys* keys, const Tm* m, const Tv* vx, const Tv* vy,
                                                const Tv* vz, const cstone::Vec3<T2> star_position, T2 star_size2,
                                                T2 removal_limit_h, T2 removal_limit_r2, T2 removal_limit_z,
                                                RemovalStatistics* device_accreted,
                                                RemovalStatistics* device_removed)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;

    // Accreted particles statistics
    RemovalStatistics accreted{};
    // Removed particles statistics
    RemovalStatistics removed{};

    if (i >= last) {}
    else
    {
        const double dx    = x[i] - star_position[0];
        const double dy    = y[i] - star_position[1];
        const double dz    = z[i] - star_position[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;

        const double r_cyl2 = (double)x[i] * x[i] + (double)y[i] * y[i];
        const double abs_z  = fabs((double)z[i]);

        if (dist2 < star_size2) { markForRemovalAndAdd(accreted, i, keys, m, vx, vy, vz); }
        else if (h[i] > removal_limit_h
                 || r_cyl2 > removal_limit_r2
                 || abs_z  > removal_limit_z) { markForRemovalAndAdd(removed, i, keys, m, vx, vy, vz); }
    }

    typedef cub::BlockReduce<RemovalStatistics, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage            temp_storage;

    RemovalStatistics block_accreted = BlockReduce(temp_storage).Sum(accreted);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAddRS(device_accreted, block_accreted); }

    RemovalStatistics block_removed = BlockReduce(temp_storage).Sum(removed);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAddRS(device_removed, block_removed); }
}

template<unsigned numThreads, typename T1, typename Th, typename Tkeys, typename T2, typename Tm, typename Tv>
__global__ void computeBinaryAccretionConditionKernel(
    size_t first, size_t last,
    const T1* x, const T1* y, const T1* z, const Th* h, Tkeys* keys,
    const Tm* m, const Tv* vx, const Tv* vy, const Tv* vz,
    cstone::Vec3<T2> star1_position, T2 star1_size2,
    T2 star1_removal_limit_h, T2 star1_removal_limit_r2, T2 star1_removal_limit_z,
    cstone::Vec3<T2> star2_position, T2 star2_size2,
    T2 star2_removal_limit_h, T2 star2_removal_limit_r2, T2 star2_removal_limit_z,
    RemovalStatistics* device_accreted1, RemovalStatistics* device_removed1,
    RemovalStatistics* device_accreted2, RemovalStatistics* device_removed2)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;

    RemovalStatistics accreted1{}, removed1{}, accreted2{}, removed2{};

    if (i < last)
    {
        const double dx1     = x[i] - star1_position[0];
        const double dy1     = y[i] - star1_position[1];
        const double dz1     = z[i] - star1_position[2];
        const double dist2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;

        const double dx2     = x[i] - star2_position[0];
        const double dy2     = y[i] - star2_position[1];
        const double dz2     = z[i] - star2_position[2];
        const double dist2_2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;

        const double r_cyl2 = (double)x[i] * x[i] + (double)y[i] * y[i];
        const double abs_z  = fabs((double)z[i]);

        if (dist2_1 < star1_size2) { markForRemovalAndAdd(accreted1, i, keys, m, vx, vy, vz); }
        else if (h[i] > star1_removal_limit_h || r_cyl2 > star1_removal_limit_r2 || abs_z > star1_removal_limit_z)
            { markForRemovalAndAdd(removed1, i, keys, m, vx, vy, vz); }

        if (dist2_2 < star2_size2) { markForRemovalAndAdd(accreted2, i, keys, m, vx, vy, vz); }
        else if (h[i] > star2_removal_limit_h || r_cyl2 > star2_removal_limit_r2 || abs_z > star2_removal_limit_z)
            { markForRemovalAndAdd(removed2, i, keys, m, vx, vy, vz); }
    }

    typedef cub::BlockReduce<RemovalStatistics, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage            temp_storage;

    RemovalStatistics block_accreted1 = BlockReduce(temp_storage).Sum(accreted1);
    __syncthreads();
    RemovalStatistics block_removed1  = BlockReduce(temp_storage).Sum(removed1);
    __syncthreads();
    RemovalStatistics block_accreted2 = BlockReduce(temp_storage).Sum(accreted2);
    __syncthreads();
    RemovalStatistics block_removed2  = BlockReduce(temp_storage).Sum(removed2);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAddRS(device_accreted1, block_accreted1);
        atomicAddRS(device_removed1,  block_removed1);
        atomicAddRS(device_accreted2, block_accreted2);
        atomicAddRS(device_removed2,  block_removed2);
    }
}

// Persistent device accumulators — zeroed before each kernel, read back after.
// Avoids per-call cudaMalloc/cudaFree (each costs 50–200 µs device-wide lock).
// Pattern matches central_force_gpu.cu (force_device, t_star_device).
static __device__ RemovalStatistics accreted_device_ss;
static __device__ RemovalStatistics removed_device_ss;

static __device__ RemovalStatistics accreted_device_bin_1;
static __device__ RemovalStatistics removed_device_bin_1;
static __device__ RemovalStatistics accreted_device_bin_2;
static __device__ RemovalStatistics removed_device_bin_2;

template<typename Treal, typename Thydro, typename Tkeys, typename Tmass>
void computeAccretionConditionGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                  const Thydro* h, Tkeys* keys, const Tmass* m, const Thydro* vx, const Thydro* vy,
                                  const Thydro* vz, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    const RemovalStatistics zero{};
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(accreted_device_ss), &zero, sizeof(zero)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(removed_device_ss),  &zero, sizeof(zero)));

    RemovalStatistics *accreted_ptr, *removed_ptr;
    checkGpuErrors(cudaGetSymbolAddress((void**)&accreted_ptr, GPU_SYMBOL(accreted_device_ss)));
    checkGpuErrors(cudaGetSymbolAddress((void**)&removed_ptr,  GPU_SYMBOL(removed_device_ss)));

    computeAccretionConditionKernel<numThreads><<<numBlocks, numThreads>>>(
        first, last, x, y, z, h, keys, m, vx, vy, vz, star.position, star.inner_size * star.inner_size,
        star.removal_limit_h, star.removal_limit_r * star.removal_limit_r, star.removal_limit_z,
        accreted_ptr, removed_ptr);

    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());

    checkGpuErrors(cudaMemcpyFromSymbol(&star.accreted_local, GPU_SYMBOL(accreted_device_ss), sizeof(star.accreted_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&star.removed_local,  GPU_SYMBOL(removed_device_ss),  sizeof(star.removed_local)));
}

template<typename Treal, typename Thydro, typename Tkeys, typename Tmass>
void computeBinaryAccretionConditionGPU(size_t first, size_t last,
                                         const Treal* x, const Treal* y, const Treal* z, const Thydro* h,
                                         Tkeys* keys, const Tmass* m, const Thydro* vx, const Thydro* vy,
                                         const Thydro* vz, StarData& star1, StarData& star2)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    const RemovalStatistics zero{};
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(accreted_device_bin_1), &zero, sizeof(zero)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(removed_device_bin_1),  &zero, sizeof(zero)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(accreted_device_bin_2), &zero, sizeof(zero)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(removed_device_bin_2),  &zero, sizeof(zero)));

    RemovalStatistics *accreted1_d, *removed1_d, *accreted2_d, *removed2_d;
    checkGpuErrors(cudaGetSymbolAddress((void**)&accreted1_d, GPU_SYMBOL(accreted_device_bin_1)));
    checkGpuErrors(cudaGetSymbolAddress((void**)&removed1_d,  GPU_SYMBOL(removed_device_bin_1)));
    checkGpuErrors(cudaGetSymbolAddress((void**)&accreted2_d, GPU_SYMBOL(accreted_device_bin_2)));
    checkGpuErrors(cudaGetSymbolAddress((void**)&removed2_d,  GPU_SYMBOL(removed_device_bin_2)));

    computeBinaryAccretionConditionKernel<numThreads><<<numBlocks, numThreads>>>(
        first, last, x, y, z, h, keys, m, vx, vy, vz,
        star1.position, star1.inner_size * star1.inner_size,
        star1.removal_limit_h, star1.removal_limit_r * star1.removal_limit_r, star1.removal_limit_z,
        star2.position, star2.inner_size * star2.inner_size,
        star2.removal_limit_h, star2.removal_limit_r * star2.removal_limit_r, star2.removal_limit_z,
        accreted1_d, removed1_d, accreted2_d, removed2_d);

    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());

    checkGpuErrors(cudaMemcpyFromSymbol(&star1.accreted_local, GPU_SYMBOL(accreted_device_bin_1), sizeof(star1.accreted_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&star1.removed_local,  GPU_SYMBOL(removed_device_bin_1),  sizeof(star1.removed_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&star2.accreted_local, GPU_SYMBOL(accreted_device_bin_2), sizeof(star2.accreted_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&star2.removed_local,  GPU_SYMBOL(removed_device_bin_2),  sizeof(star2.removed_local)));
}

#define COMPUTE_ACCRETION_CONDITION_GPU(Treal, Thydro, Tkeys, Tmass)                                                   \
    template void computeAccretionConditionGPU(size_t first, size_t last, const Treal* x, const Treal* y,              \
                                               const Treal* z, const Thydro* h, Tkeys* keys, const Tmass* m,           \
                                               const Thydro* vx, const Thydro* vy, const Thydro* vz, StarData& star);

COMPUTE_ACCRETION_CONDITION_GPU(double, double, size_t, double);
COMPUTE_ACCRETION_CONDITION_GPU(double, float, size_t, double);
COMPUTE_ACCRETION_CONDITION_GPU(double, float, size_t, float);

#define COMPUTE_BINARY_ACCRETION_CONDITION_GPU(Treal, Thydro, Tkeys, Tmass)                                         \
    template void computeBinaryAccretionConditionGPU(size_t, size_t,                                                  \
                                                      const Treal* x, const Treal* y, const Treal* z,               \
                                                      const Thydro* h, Tkeys* keys, const Tmass* m,                  \
                                                      const Thydro* vx, const Thydro* vy, const Thydro* vz,          \
                                                      StarData&, StarData&);

COMPUTE_BINARY_ACCRETION_CONDITION_GPU(double, double, size_t, double);
COMPUTE_BINARY_ACCRETION_CONDITION_GPU(double, float, size_t, double);
COMPUTE_BINARY_ACCRETION_CONDITION_GPU(double, float, size_t, float);

} // namespace disk
