//
// Created by Noah Kubli on 11.03.2024.
//

#include <thrust/functional.h>

#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"

#include "central_force_gpu.hpp"
#include "central_potential.hpp"
#include "star_data.hpp"

namespace disk
{

static __device__ cstone::Vec4<double> force_device;
static __device__ float                t_star_device;

static __device__ cstone::Vec4<double> force_device_1;
static __device__ float                t_star_device_1;
static __device__ cstone::Vec4<double> force_device_2;
static __device__ float                t_star_device_2;

template<typename T>
__device__ void atomicAddVec4(cstone::Vec4<T>* x, const cstone::Vec4<T>& y)
{
    atomicAdd(&(*x)[0], y[0]);
    atomicAdd(&(*x)[1], y[1]);
    atomicAdd(&(*x)[2], y[2]);
    atomicAdd(&(*x)[3], y[3]);
}

template<size_t numThreads, typename Data>
__global__ void computeCentralForceGPUKernel(size_t first, size_t last, const Data d, StarPotentialType potentialType)
{
    cstone::LocalIndex   i = first + blockDim.x * blockIdx.x + threadIdx.x;
    cstone::Vec4<double> force{};
    float                t_star{INFINITY};

    if (i >= last) { force = {0., 0., 0., 0.}; }
    else
    {
        if (potentialType == StarPotentialType::newtonian) { newtonianGravity(d, i, force, t_star); }
        else if (potentialType == StarPotentialType::einstein_precession) { einsteinPrecession(d, i, force, t_star); }
    }

    typedef cub::BlockReduce<cstone::Vec4<double>, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage               temp_storage;

    cstone::Vec4<double> force_block = BlockReduce(temp_storage).Sum(force);
    __syncthreads();

    typedef cub::BlockReduce<float, numThreads>       BlockReduceTStar;
    __shared__ typename BlockReduceTStar::TempStorage temp_storage_t_star;
    BlockReduceTStar                                  reduce_t_star(temp_storage_t_star);

    float t_star_block = reduce_t_star.Reduce(t_star, thrust::minimum<>{});
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAddVec4(&force_device, force_block);
        cstone::atomicMinFloat(&t_star_device, t_star_block);
    }
}

template<size_t numThreads, typename Data>
__global__ void computeBinaryCentralForceGPUKernel(size_t first, size_t last,
                                                    const Data data1, const Data data2,
                                                    StarPotentialType potentialType)
{
    cstone::LocalIndex   i      = first + blockDim.x * blockIdx.x + threadIdx.x;
    cstone::Vec4<double> force1{}, force2{};
    float                t_star1{INFINITY}, t_star2{INFINITY};

    if (i >= last) { force1 = {0., 0., 0., 0.}; force2 = {0., 0., 0., 0.}; }
    else
    {
        if (potentialType == StarPotentialType::newtonian)
        {
            newtonianGravity(data1, i, force1, t_star1);
            newtonianGravity(data2, i, force2, t_star2);
        }
        else if (potentialType == StarPotentialType::einstein_precession)
        {
            einsteinPrecession(data1, i, force1, t_star1);
            einsteinPrecession(data2, i, force2, t_star2);
        }
    }

    typedef cub::BlockReduce<cstone::Vec4<double>, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage               temp_storage;

    cstone::Vec4<double> force1_block = BlockReduce(temp_storage).Sum(force1);
    __syncthreads();
    cstone::Vec4<double> force2_block = BlockReduce(temp_storage).Sum(force2);
    __syncthreads();

    typedef cub::BlockReduce<float, numThreads>       BlockReduceTStar;
    __shared__ typename BlockReduceTStar::TempStorage temp_storage_t;
    BlockReduceTStar                                  reducer_t(temp_storage_t);

    float t_star1_block = reducer_t.Reduce(t_star1, thrust::minimum<>{});
    __syncthreads();
    float t_star2_block = reducer_t.Reduce(t_star2, thrust::minimum<>{});
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAddVec4(&force_device_1, force1_block);
        atomicAddVec4(&force_device_2, force2_block);
        cstone::atomicMinFloat(&t_star_device_1, t_star1_block);
        cstone::atomicMinFloat(&t_star_device_2, t_star2_block);
    }
}

template<typename Treal, typename Thydro, typename Tmass>
void computeCentralForceGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,
                            Thydro* ay, Thydro* az, const Tmass* m, Treal g, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    cstone::Vec4<double> force_local{0., 0., 0., 0.};
    float                t_star_local{INFINITY};
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(force_device), &force_local, sizeof(force_local)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(t_star_device), &t_star_local, sizeof(t_star_local)));

    const double         grav_softening2 = star.grav_softening * star.grav_softening;
    CentralPotentialData data{x, y, z, m, ax, ay, az, g, star.m, grav_softening2, 1.0};
    data.star_position = star.position; // Initializing in aggregate list produces an error
    if (last > first)
    {
        computeCentralForceGPUKernel<numThreads><<<numBlocks, numThreads>>>(first, last, data, star.potentialType);

        checkGpuErrors(cudaDeviceSynchronize());
        checkGpuErrors(cudaGetLastError());
    }
    checkGpuErrors(cudaMemcpyFromSymbol(&force_local, GPU_SYMBOL(force_device), sizeof(force_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&t_star_local, GPU_SYMBOL(t_star_device), sizeof(t_star_local)));

    star.force_local = force_local;
    star.t_star      = t_star_local;
}

template<typename Treal, typename Thydro, typename Tmass>
void computeBinaryCentralForceGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                   Thydro* ax, Thydro* ay, Thydro* az, const Tmass* m, Treal g,
                                   StarData& star1, StarData& star2)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    cstone::Vec4<double> zero_force{0., 0., 0., 0.};
    float                inf_t{INFINITY};
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(force_device_1), &zero_force, sizeof(zero_force)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(t_star_device_1), &inf_t, sizeof(inf_t)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(force_device_2), &zero_force, sizeof(zero_force)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(t_star_device_2), &inf_t, sizeof(inf_t)));

    const double         grav_softening2_1 = star1.grav_softening * star1.grav_softening;
    const double         grav_softening2_2 = star2.grav_softening * star2.grav_softening;

    CentralPotentialData data1{x, y, z, m, ax, ay, az, g, star1.m, grav_softening2_1, 1.0};
    data1.star_position = star1.position;
    CentralPotentialData data2{x, y, z, m, ax, ay, az, g, star2.m, grav_softening2_2, 1.0};
    data2.star_position = star2.position;

    if (last > first)
    {
        computeBinaryCentralForceGPUKernel<numThreads><<<numBlocks, numThreads>>>(
            first, last, data1, data2, star1.potentialType);

        checkGpuErrors(cudaDeviceSynchronize());
        checkGpuErrors(cudaGetLastError());
    }

    cstone::Vec4<double> force1_local, force2_local;
    float                t_star1_local, t_star2_local;
    checkGpuErrors(cudaMemcpyFromSymbol(&force1_local, GPU_SYMBOL(force_device_1), sizeof(force1_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&t_star1_local, GPU_SYMBOL(t_star_device_1), sizeof(t_star1_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&force2_local, GPU_SYMBOL(force_device_2), sizeof(force2_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&t_star2_local, GPU_SYMBOL(t_star_device_2), sizeof(t_star2_local)));

    star1.force_local = force1_local;
    star1.t_star      = t_star1_local;
    star2.force_local = force2_local;
    star2.t_star      = t_star2_local;
}

#define COMPUTE_CENTRAL_FORCE_GPU(Treal, Thydro, Tmass)                                                                \
    template void computeCentralForceGPU(size_t, size_t, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,   \
                                         Thydro* ay, Thydro* az, const Tmass* m, Treal g, StarData&);

COMPUTE_CENTRAL_FORCE_GPU(double, double, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, float);

#define COMPUTE_BINARY_CENTRAL_FORCE_GPU(Treal, Thydro, Tmass)                                                       \
    template void computeBinaryCentralForceGPU(size_t, size_t, const Treal* x, const Treal* y, const Treal* z,       \
                                               Thydro* ax, Thydro* ay, Thydro* az, const Tmass* m, Treal g,          \
                                               StarData&, StarData&);

COMPUTE_BINARY_CENTRAL_FORCE_GPU(double, double, double);
COMPUTE_BINARY_CENTRAL_FORCE_GPU(double, float, double);
COMPUTE_BINARY_CENTRAL_FORCE_GPU(double, float, float);

} // namespace disk
