/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Simulation data initialization from an HDF5 file
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <vector>

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"

#include "isim_init.hpp"

namespace sphexa
{

/*! @brief May this conserved field be absent from a checkpoint and reconstructed instead of read?
 *
 *  Two groups qualify:
 *   - Accumulator diagnostics ("du_cool_accum"): they measure a window between two dumps, so a restart
 *     legitimately starts them from zero.
 *   - Integrator state the lean dump omits ("m", "x_m1/y_m1/z_m1", "du_m1"): these are not written on
 *     every dump because they are exactly recoverable -- m from the "particleMass" root scalar, x_m1
 *     from the stored velocity, du_m1 as a half-step that self-corrects after one step. When absent they
 *     are rebuilt by reconstructRestartState(), not left at zero.
 *
 *  Every other conserved field describes physical state that cannot be recovered, so its absence is a
 *  hard error rather than a silent zero (a missing "vx" left at zero would look like a valid run).
 */
inline bool isOptionalOnRestore(const std::string& key)
{
    return key == "du_cool_accum" || key == "m" || key == "x_m1" || key == "y_m1" || key == "z_m1" ||
           key == "du_m1";
}

/*! @brief Restore all conserved fields of @p d from the current step; return the ones that were absent.
 *
 *  A field listed by isOptionalOnRestore that is missing from the file is left zero-initialized here and
 *  its name returned, so the caller can reconstruct it. A missing non-optional field throws.
 */
template<class Dataset>
std::vector<std::string> restoreDataset(IFileReader* reader, Dataset& d)
{
    d.loadOrStoreAttributes(reader);
    d.resize(reader->localNumParticles());

    std::vector<std::string> missingFields;

    // Datasets actually present in this step. Checking membership up front lets an absent optional field
    // be skipped without ever calling readField on it -- otherwise H5hut prints a "Cannot open dataset"
    // error to stderr from deep inside the failed read, even though the absence is expected and handled.
    // Empty (readers that don't implement stepDatasets) -> fall back to the read-and-catch path below.
    std::vector<std::string> present = reader->stepDatasets();
    auto isPresent = [&present](const std::string& key)
    { return present.empty() || std::find(present.begin(), present.end(), key) != present.end(); };

    auto fieldPointers = d.data();
    for (size_t i = 0; i < fieldPointers.size(); ++i)
    {
        if (d.isConserved(i))
        {
            const std::string key = d.fieldNames[i];

            if (!isPresent(Dataset::prefix + key))
            {
                if (!isOptionalOnRestore(key))
                {
                    throw std::runtime_error("Cannot restart: required field '" + key +
                                             "' is missing from this step and cannot be reconstructed.\n");
                }
                // Zero the allocated buffer so a field that is reconstructed later from zero (du_m1, the
                // du_cool_accum accumulator) has a defined starting value rather than leftover memory.
                std::visit([](auto field)
                           { *field = std::decay_t<decltype(*field)>(field->size()); },
                           fieldPointers[i]);
                missingFields.push_back(key);
                if (reader->rank() == 0) { std::cout << "restoring " << key << " (absent; will reconstruct)\n"; }
                continue;
            }

            if (reader->rank() == 0) { std::cout << "restoring " << key; }
            auto t0 = std::chrono::high_resolution_clock::now();
            std::visit(
                [reader, key](auto field)
                {
                    using T = std::remove_reference<decltype(*field->data())>::type;
                    std::vector<T> tmp(field->size());
                    reader->readField(Dataset::prefix + key, tmp.data());
                    *field = std::move(tmp);
                },
                fieldPointers[i]);
            MPI_Barrier(MPI_COMM_WORLD);
            auto  t1       = std::chrono::high_resolution_clock::now();
            int   typeSize = std::visit([](auto field) { return sizeof(*field->data()); }, fieldPointers[i]);
            float readTime = std::chrono::duration<float>(t1 - t0).count();
            if (reader->rank() == 0)
            {
                float sizeGB = float(typeSize) * reader->globalNumParticles() / 1024 / 1024 / 1024;
                std::cout << ", " << sizeGB << " GB in " << readTime << " s, " << sizeGB / readTime << " GB/s"
                          << std::endl;
            }
        }
    }
    return missingFields;
}

/*! @brief Rebuild the integrator state a lean dump omits, from the fields it does carry.
 *
 *  Called after restoreDataset for the hydro dataset. @p missing is the set of conserved fields that were
 *  absent from the file (from restoreDataset); only those are reconstructed, so restarting from an old
 *  full checkpoint that still contains them is a no-op.
 *
 *  Reconstruction rules (see BinaryProp::conservedFields and sph::positionUpdate):
 *   - m        <- the uniform "particleMass" root attribute. Errors out if the array is missing *and* the
 *                 scalar is absent, rather than running with zero masses.
 *   - x_m1     <- v * minDt. Exact relation is x_m1 = v*minDt - 0.5*a*minDt^2; the dropped acceleration
 *                 term is ~1e-4 of x_m1, a one-time restart perturbation far below per-step truncation.
 *   - du_m1    <- 0. The leapfrog energy step reuses it once, then overwrites it with the freshly
 *                 computed du, so the transient lasts a single step.
 *
 *  Works on CPU and GPU via the toHost() / move-assign (H2D) idiom used by the analytic initializers.
 */
template<class HydroData>
void reconstructRestartState(IFileReader* reader, HydroData& d, const std::vector<std::string>& missing)
{
    auto absent = [&missing](const std::string& k)
    { return std::find(missing.begin(), missing.end(), k) != missing.end(); };

    const size_t n = d.x.size();

    if (absent("m"))
    {
        double particleMass = 0.;
        try
        {
            reader->fileAttribute("particleMass", &particleMass, 1);
        }
        catch (std::out_of_range&)
        {
            throw std::runtime_error(
                "Cannot restart: per-particle 'm' is absent from this step and no 'particleMass' root "
                "attribute is present to reconstruct it from.\n");
        }
        using MType = std::decay_t<decltype(d.m)>::value_type;
        std::vector<MType> m(n, MType(particleMass));
        d.m = std::move(m);
    }

    if (absent("x_m1") || absent("y_m1") || absent("z_m1"))
    {
        using XM1Type = std::decay_t<decltype(d.x_m1)>::value_type;
        auto&& vx = toHost(d.vx);
        auto&& vy = toHost(d.vy);
        auto&& vz = toHost(d.vz);
        std::vector<XM1Type> x_m1(n), y_m1(n), z_m1(n);
        const double         dt = d.minDt;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            x_m1[i] = XM1Type(vx[i] * dt);
            y_m1[i] = XM1Type(vy[i] * dt);
            z_m1[i] = XM1Type(vz[i] * dt);
        }
        d.x_m1 = std::move(x_m1);
        d.y_m1 = std::move(y_m1);
        d.z_m1 = std::move(z_m1);
    }

    // du_m1 absent -> already zero-initialized by restoreDataset, which is the intended reconstruction.

    // du_cool_accum is a per-dump *window* accumulator (cooling energy lost since the previous dump),
    // reset in memory right after each write. But it is also written to disk, so on a restart it is
    // reloaded carrying the last pre-restart window's value -- the first post-restart dump then
    // double-counts it (~2x). Zero it here so every restart begins a fresh cooling window, mirroring the
    // star-accumulator reset in BinaryProp::load(). Unconditional (not gated on `absent`): the field is
    // present in a lean dump and reloaded with a stale value, so it must be cleared even when found.
    // No-op for propagators that don't allocate it (size 0).
    if (d.du_cool_accum.size() == n && n > 0)
    {
        using CoolType = std::decay_t<decltype(d.du_cool_accum)>::value_type;
        d.du_cool_accum = std::vector<CoolType>(n, CoolType(0));
    }
}

template<class SimulationData>
auto restoreData(IFileReader* reader, SimulationData& simData)
{
    using T = typename SimulationData::RealType;

    cstone::Box<T> box(0, 1);
    box.loadOrStore(reader);

    // Lean dumps omit reconstructable integrator state (m, x_m1/y_m1/z_m1, du_m1); rebuild whatever the
    // hydro step didn't carry. Only the hydro dataset has these fields. Restoring from an old full
    // checkpoint returns no missing fields, so this is a no-op there.
    auto missingHydro = restoreDataset(reader, simData.hydro);
    reconstructRestartState(reader, simData.hydro, missingHydro);

    restoreDataset(reader, simData.chem);

    return box;
}

template<class Dataset>
class FileInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          initStep = -1;

public:
    explicit FileInit(const std::string& fname, int initStep_, IFileReader* reader)
        : h5_fname(fname)
        , initStep(initStep_)
    {
        // Read file attributes and put them in settings_ such that they propagate to the new output after a restart
        readFileAttributes(settings_, h5_fname, reader, false);
    }

    cstone::Box<typename Dataset::RealType> init(int /*rank*/, int numRanks, size_t /*n*/, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        reader->setStep(h5_fname, initStep, FileMode::collective);
        auto box = restoreData(reader, simData);
        reader->closeStep();
        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

template<class Dataset>
class FileSplitInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          numSplits;

public:
    explicit FileSplitInit(const std::string& fname, int numSplits_, IFileReader* reader)
        : h5_fname(fname)
        , numSplits(numSplits_)
    {
        if (numSplits < 1)
        {
            throw std::runtime_error("Number of particle splits must be a positive integer. Provided value: " +
                                     std::to_string(numSplits));
        }
        // Read file attributes and put them in constants_ such that they propagate to the new output after a restart
        readFileAttributes(settings_, h5_fname, reader, false);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int, size_t, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        constexpr bool gpu = cstone::HaveGpu<typename Dataset::AcceleratorType>{};
        reader->setStep(h5_fname, -1, FileMode::collective);

        size_t numParticlesInFile = reader->localNumParticles();
        size_t numParticlesSplit  = numParticlesInFile * numSplits;

        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        cstone::Box<T> box(0, 1);
        box.loadOrStore(reader);

        auto& d = simData.hydro;
        d.loadOrStoreAttributes(reader);

        d.numParticlesGlobal = reader->globalNumParticles() * numSplits;
        d.iteration          = 1;
        d.ttot               = 0.0;
        d.minDt /= (100 * numSplits);
        d.minDt_m1 /= (100 * numSplits);

        d.x.resize(numParticlesSplit);
        d.y.resize(numParticlesSplit);
        d.z.resize(numParticlesSplit);
        d.h.resize(numParticlesSplit);

        std::vector<cstone::LocalIndex> sfcOrder(numParticlesInFile);
        {
            std::vector<T> x0(numParticlesInFile), y0(numParticlesInFile), z0(numParticlesInFile),
                tmp(numParticlesInFile);
            reader->readField("x", x0.data());
            reader->readField("y", y0.data());
            reader->readField("z", z0.data());

            std::vector<KeyType> keys(numParticlesInFile);
            cstone::computeSfcKeys(x0.data(), y0.data(), z0.data(), cstone::sfcKindPointer(keys.data()),
                                   numParticlesInFile, box);
            std::iota(sfcOrder.begin(), sfcOrder.end(), 0);
            cstone::sort_by_key(keys.begin(), keys.end(), sfcOrder.begin());

            auto gatherSwap = [&tmp](auto& v, auto& order)
            {
                cstone::gather<cstone::LocalIndex>(order, v.data(), tmp.data());
                swap(v, tmp);
            };
            gatherSwap(x0, sfcOrder);
            gatherSwap(y0, sfcOrder);
            gatherSwap(z0, sfcOrder);

            std::vector<T> x(numParticlesSplit);
            std::vector<T> y(numParticlesSplit);
            std::vector<T> z(numParticlesSplit);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numParticlesInFile; ++i)
            {
                size_t sIdx = numSplits * i;

                x[sIdx] = x0[i];
                y[sIdx] = y0[i];
                z[sIdx] = z0[i];

                bool isLast   = (i == numParticlesInFile - 1);
                long keyDelta = (isLast ? -(keys[i] - keys[i - 1]) : keys[i + 1] - keys[i]) / (numSplits + isLast);

                for (size_t j = 1; j < numSplits; ++j)
                {
                    auto [ixj, iyj, izj] = cstone::decodeSfc(cstone::sfcKey(keys[i] + j * keyDelta));

                    x[sIdx + j] = box.xmin() + (ixj * box.lx()) / cstone::maxCoord<KeyType>{};
                    y[sIdx + j] = box.ymin() + (iyj * box.ly()) / cstone::maxCoord<KeyType>{};
                    z[sIdx + j] = box.zmin() + (izj * box.lz()) / cstone::maxCoord<KeyType>{};
                }
            }
            d.x = std::move(x);
            d.y = std::move(y);
            d.z = std::move(z);
        }

        auto replicateField = [&sfcOrder, numParticlesInFile, numParticlesSplit,
                               this](IFileReader* reader, const std::string& key, auto& dest, T scale)
        {
            std::vector<T> src(numParticlesInFile), tmp(numParticlesInFile);
            reader->readField(key, src.data());
            cstone::gather<cstone::LocalIndex>(sfcOrder, src.data(), tmp.data());
            swap(src, tmp);
            tmp.clear();

            using DestVectorType = std::decay_t<decltype(dest)>::value_type;
            std::vector<DestVectorType> outTmp(numParticlesSplit);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numParticlesInFile; ++i)
            {
                size_t sIdx = numSplits * i;
                std::fill(outTmp.data() + sIdx, outTmp.data() + sIdx + numSplits, src[i] * scale);
            }
            dest = std::move(outTmp);
        };

        d.resize(numParticlesSplit);
        replicateField(reader, "m", d.m, T(1) / numSplits);
        replicateField(reader, "h", d.h, T(1) / std::cbrt(numSplits));
        replicateField(reader, "vx", d.vx, T(1));
        replicateField(reader, "vy", d.vy, T(1));
        replicateField(reader, "vz", d.vz, T(1));
        replicateField(reader, "temp", d.temp, T(1));
        cstone::fill<gpu>(d.du_m1.begin(), d.du_m1.end(), 0);
        cstone::fill<gpu>(d.rung.begin(), d.rung.end(), 0);
        cstone::scaleGpuAcc<gpu>(d.vx.data(), d.vx.data() + d.vx.size(), d.x_m1.data(), d.minDt);
        cstone::scaleGpuAcc<gpu>(d.vy.data(), d.vy.data() + d.vy.size(), d.y_m1.data(), d.minDt);
        cstone::scaleGpuAcc<gpu>(d.vz.data(), d.vz.data() + d.vz.size(), d.z_m1.data(), d.minDt);

        if (d.isAllocated("alpha"))
        {
            try
            {
                replicateField(reader, "alpha", d.alpha, T(1));
            }
            catch (std::runtime_error&)
            {
                cstone::fill<gpu>(d.alpha.begin(), d.alpha.end(), d.alphamin);
            }
        }

        reader->closeStep();

        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa
