/*! @file
 * @brief A Propagator class for binary systems of massive objects
 *
 * @author Federico Ragno
 */
#pragma once

#include <cstdint>
#include <cstdio>

#include "cstone/fields/field_get.hpp"
#include "io/arg_parser.hpp"
#include "ipropagator.hpp"
#include "std_hydro.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "accretion.hpp"
#include "beta_cooling.hpp"
#include "binary_energy.hpp"
#include "central_force.hpp"
#include "exchange_star_position.hpp"
#include "star_data.hpp"
#include "reciprocal_force.hpp"
#include "binary_data.hpp"


namespace sphexa
{

using namespace sph;
using util::FieldList;

template<class DomainType, class DataType>
class BinaryProp : public HydroProp<DomainType, DataType>
{
protected:
    using Base = HydroProp<DomainType, DataType>;
    using Base::timer;

    using T = typename DataType::RealType;

    disk::StarData star1;
    disk::StarData star2;

    /*! @brief Mass/energy of particles that left the simulation without being accreted, accumulated since
     *  the last dump. Both channels are star-independent -- neither the h/r/z cutoffs nor neighbor-search
     *  convergence reference a star position -- so they are global, not star1::/star2:: attributes.
     *  Kept as two separate channels because they are distinct failure modes worth telling apart.
     */
    double limitRemovedMassAccum_{0.};
    double limitRemovedEnergyAccum_{0.};
    double nonconvergedRemovedMassAccum_{0.};
    double nonconvergedRemovedEnergyAccum_{0.};

    /*! @brief Binary-only particle fields, layered on top of HydroProp's lists.
     *
     *  du_cool_accum MUST be conserved, not dependent: it accumulates the beta-cooling energy loss across
     *  every iteration between two dumps, so it has to survive domain sync and be gathered/reordered along
     *  with the particles it belongs to. Dependent fields are explicitly reused as sync scratch space,
     *  which silently shredded this accumulator (garbage index data appeared in the output).
     *
     *  ugrav stays dependent -- the gravity traversal recomputes it from zero every iteration (see the
     *  zeroing in HydroProp::computeForces), so it needs no cross-iteration persistence. It only has to be
     *  allocated, which setDependent below achieves.
     *
     *  Both are declared here rather than in HydroProp so plain hydro cases, DiskProp etc. neither
     *  allocate nor sync them.
     */
    using BinaryConservedFields = FieldList<"du_cool_accum">;
    using BinaryDependentFields = FieldList<"ugrav">;

public:
    BinaryProp(std::ostream& output, size_t rank, const InitSettings& settings) // Settings are not used!
        : Base(output, rank)
    {
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        // Read star position from hdf5 File

        //FR: strips runtime modifiers separated by ":" from input path. Eg: if initCond is "disk.h5:10", path is "disk.h5"
        std::string path = removeModifiers(initCond);

        //FR: checks whether file associated with path exists. In test cases (eg: initCond = "evrard") star data will not be loaded because it doesn't come from a file
        if (std::filesystem::exists(path))
        {
            /*FR: Determines which snapshot in the h5 file to read initial conditions from.
            Initial conditions file format:
                disk.h5
                ├── Step#0
                ├── Step#1
                ├── Step#2
                ...
                ├── Step#10
            If initCont = "disk.h5:10" initial conditions will be taken from 10th timestep
            */
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);

            star1.loadOrStoreAttributes(reader, "star1");
            star2.loadOrStoreAttributes(reader, "star2");

            // A restart must not resume mid-window with stale accumulated diagnostics read back
            // from the file -- always start a fresh accumulation window on load.
            star1.resetAccumulatedDiagnostics();
            star2.resetAccumulatedDiagnostics();

            reader->closeStep();

            //FR Ensure binary is physically consistent
            if( ! binary_is_consistent(star1, star2) ) { 
                throw std::runtime_error("Binary propagator initialization error: star binary is not consistent" );
            }

            std::printf("star 1 position: %lf\t%lf\t%lf\n", star1.position[0], star1.position[1], star1.position[2]);
            std::printf("star 1 mass: %lf\n", star1.m);
            std::printf("star 2 position: %lf\t%lf\t%lf\n", star2.position[0], star2.position[1], star2.position[2]);
            std::printf("star 2 mass: %lf\n", star2.m);

        }
    }

    /*! @brief Fields written on every dump. Every dump is identical and every dump is restartable.
     *
     *  Deliberately *not* Base::conservedFields(): that list is the integrator's full state, and three
     *  parts of it are dead weight on disk because they are exactly recoverable at load time (see
     *  ParticlesData::reconstructRestartState):
     *   - m         is uniform in these setups, so the single value is written once as the "particleMass"
     *               root attribute and the per-particle array reconstructed on restart.
     *   - x_m1/y_m1/z_m1 are integrator state, not physics: x_m1 = vx*minDt - 0.5*ax*minDt^2, i.e. vx to
     *               ~1e-4, and vx is stored. They are reconstructed from vx on restart.
     *   - du_m1     likewise integrator half-step history; reconstructed as 0 (exact again after one step).
     *
     *  These fields are still *allocated* (they remain in HydroProp's conserved list, so the integrator
     *  has them in memory) -- they are just not written to disk. Because they are reconstructed rather
     *  than required, a plain intermediate dump is a complete restart point, so an interrupted run
     *  resumes from its actual last dump with no lost segment.
     */
    std::vector<std::string> conservedFields() const override
    {
        return {"x", "y", "z", "h", "u", "vx", "vy", "vz", "id", "ugrav", "du_cool_accum"};
    }

    /*! @brief Fields stored at reduced width on disk.
     *
     *  The simulation keeps these in full precision internally; only the written copy is narrowed, so the
     *  error is a single bounded rounding at write time rather than something that compounds. float32
     *  gives ~2e-6 absolute on positions spanning +-50, orders of magnitude below the smoothing length,
     *  and 6e-8 relative on u. id fits comfortably in uint32 (<= 5e5 particles), so that one is lossless.
     */
    static std::vector<std::string> narrowOutputFields() { return {"x", "y", "z", "u", "du_cool_accum", "id"}; }

    //! @brief HydroProp's fields, plus the binary-only ones (see BinaryConservedFields).
    void activateFields(DataType& simData) override
    {
        Base::activateFields(simData);

        auto& d = simData.hydro;
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(BinaryConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(BinaryDependentFields{}));
    }

    /*! @brief Same as HydroProp::sync, but carrying the binary-only fields through the exchange.
     *
     *  Mirrors Base::sync exactly; the conserved tuple is extended with BinaryConservedFields so
     *  du_cool_accum is gathered/reordered with its particles instead of being treated as scratch.
     */
    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;

        /*! Only the conserved tuple is extended. The scratch tuple must stay exactly as HydroProp passes
         *  it, because cstone::Domain::sync/syncGrav treat its **last** element specially: it is handed to
         *  SfcSorter as the SFC ordering buffer (see domain.hpp, `std::get<sizeof...(Vectors2) - 1>`, and
         *  note staticChecks validates only discardLastElement(scratch)). Upstream that slot is "nc", a
         *  uint32 index buffer. Appending anything after it silently redesignates that field as the sort
         *  buffer -- which is what made the sorter write particle indices into a float ugrav array and
         *  crash the GPU run with an illegal access, and, in an earlier revision where du_cool_accum sat
         *  last, is what filled it with consecutive integer pairs.
         *
         *  ugrav is deliberately absent here: it needs to be *allocated* (setDependent in activateFields,
         *  so d.resize sizes it), but it must not join the exchange. The gravity traversal rebuilds it
         *  from zero every iteration after sync, so it has nothing worth carrying across.
         *
         *  du_cool_accum is double; the scratch list contains "du" (also double), which satisfies
         *  staticChecks' requirement that some scratch buffer be at least as wide as each conserved field.
         *
         *  std::tie, not get<BinaryConservedFields>(d): get<> on a single-element FieldList yields a bare
         *  reference rather than a 1-tuple, which tuple_cat won't take.
         */
        auto conserved = std::tuple_cat(get<typename Base::ConservedFields>(d), std::tie(get<"du_cool_accum">(d)));
        auto dependent = get<typename Base::DependentFields>(d);

        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            conserved, dependent);
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), conserved), dependent);
        }
        d.treeView = domain.octreeProperties();
    }

    void save(IFileWriter* writer) override
    {
        star1.loadOrStoreAttributes(writer, "star1");
        star2.loadOrStoreAttributes(writer, "star2");

        // Global (not per-star) accumulators: neither removal channel is attributable to a specific star.
        writer->stepAttribute("removed_limit_mass_accum", &limitRemovedMassAccum_, 1);
        writer->stepAttribute("removed_limit_energy_accum", &limitRemovedEnergyAccum_, 1);
        writer->stepAttribute("removed_nonconverged_mass_accum", &nonconvergedRemovedMassAccum_, 1);
        writer->stepAttribute("removed_nonconverged_energy_accum", &nonconvergedRemovedEnergyAccum_, 1);

        // The just-written accumulators cover [previous dump, this dump]. integrate() runs after
        // save() in the main loop, so resetting here means the next window starts clean and this
        // iteration's own accretion/cooling (computed later, in integrate()) is never lost or
        // double-counted across the dump boundary.
        star1.resetAccumulatedDiagnostics();
        star2.resetAccumulatedDiagnostics();
        limitRemovedMassAccum_          = 0.;
        limitRemovedEnergyAccum_        = 0.;
        nonconvergedRemovedMassAccum_   = 0.;
        nonconvergedRemovedEnergyAccum_ = 0.;
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        // Not Base::saveFields: the binary output narrows selected fields on disk (see
        // narrowOutputFields) while the simulation keeps full precision in memory.
        Base::outputAllocatedFields(writer, simData, narrowOutputFields());

        /*! The per-particle m array is dropped from intermediate dumps (it is uniform in these setups),
         *  so publish the scalar once at the file root -- otherwise nothing downstream could recover
         *  masses. Reduced with MAX rather than read locally because fileAttribute, unlike stepAttribute,
         *  does not broadcast rank 0's value, and a rank owning no particles would contribute 0.
         */
        {
            auto& dm = simData.hydro.m;
            using MVec = std::decay_t<decltype(dm)>;
            typename MVec::value_type localMass{};
            if (last > first)
            {
                if constexpr (IsDeviceVector<MVec>{}) { memcpyD2H(rawPtr(dm) + first, 1, &localMass); }
                else { localMass = dm[first]; }
            }
            double particleMass = double(localMass);
            MPI_Allreduce(MPI_IN_PLACE, &particleMass, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            writer->fileAttribute("particleMass", &particleMass, 1);
        }

        // Reset the per-particle beta-cooling accumulator right after it's written, so the next
        // window starts clean (mirrors the StarData accumulator reset in save(), which runs
        // immediately after this in the same dump). Must work on both CPU (plain vector) and GPU
        // (device vector, no host-side operator[]) -- same device-aware fill idiom as
        // HydroProp::zeroKeys.
        auto& d = simData.hydro;
        cstone::fill<IsDeviceVector<std::decay_t<decltype(d.du_cool_accum)>>{}>(
            rawPtr(d.du_cool_accum) + first, rawPtr(d.du_cool_accum) + last, 0.0);
    }

    /* First key function in binarypropagator. Includes:
    In Hydroprop
        1) Domain synchronization
        2) Density evaluation
        3) Thermodynamic parameters evaluation (P, c, T) via chosen EOS
        4) Evaluation and writing of: acceleration; energy; courant-limited timestep
    In binaryprop
        5) Beta cooling
        6) Binary-disk; binary-binary forces
    */
    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);               // compute hydro forces

        auto&        d     = simData.hydro;                 //FR: alias to hydrodynamical particle fields
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();

        // applyDu=true, accumulateLoss=false: add the cooling rate to du here (feeds the du timestep
        // limiter). The energy-loss accumulation is deferred to integrate() after computeTimestep, so
        // it multiplies by the current step's dt rather than the previous step's (see below).
        disk::betaCoolingBinary(first, last, d, star1, star2, /*applyDu=*/true, /*accumulateLoss=*/false);

        timer.step("betaCooling");

        disk::computeBinaryCentralForce(first, last, d, star1, star2); //FR: computes force generated by central object, affecting acceleration of particles
        timer.step("computeCentralForce");

        // TODO (not stringent): einstein precession
        disk::computeBinaryForce(star1, star2, d);          //FR computes the forces between the stars
        timer.step("computeBinaryForce");
    }

    /* Second key function of binarypropagator. Includes:
        1) Timestep computation
        2) Energy floor implementation
        3) Smoothing length calculation (enforcing target number of neighbors)
        4) Force calculation + Position and velocity integration
        5) Accretion + removal of particles
    */
    void integrate(DomainType& domain, DataType& simData) override
    {
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();
        auto&        d     = simData.hydro;
        
        /*FR:
        Compute a maximal timestep based on how fast the internal energy changes.
        Writes: star.t_du
        */
        disk::duTimestep(first, last, d, star1);
        star2.t_du = star1.t_du;
        timer.step("duTimestep");
        
        /*FR:
        Computes global d.minDT with the standard SPH limits plus two extra limits coming from the disk:
            1) star.t_du: cooling rate limit
            2) d.etaAcc * star.t_star: central force acceleration limit
        
        Selects most restrictive physical timescale out of:
            1) CFL                      dt ~ dx / (c_s + v_signal)
            2) acceleration-based       dt ~ a / a_dot
            3) density-based            dt ~ rho / rho_dot
            4) energy-based             dt ~ u / u_dot
            5) gravity-based            dt ~ sqrt(r_min^3 / GM)
        
        TODO (not stringent): need to add a binary orbital period!
        */
        computeTimestep(first, last, d, 
            star1.t_du,
            d.etaAcc * star1.t_star,
            d.etaAcc * star2.t_star
        );
        timer.step("Timestep");

        // Now that computeTimestep has set d.minDt to the CURRENT step's dt, record the beta-cooling
        // energy loss = du_cool * d.minDt. Runs before computePositions, so u/rho/positions still hold
        // this step's values and du_cool recomputes identically to the applyDu pass in computeForces --
        // only the (now correct) dt differs. Fixes a ~2% systematic from using the previous step's dt.
        disk::betaCoolingBinary(first, last, d, star1, star2, /*applyDu=*/false, /*accumulateLoss=*/true);
        timer.step("betaCoolingLoss");

        //FR: Apply energy floor to avoid minDt going to 0
        computePositions(Base::groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        {
            uint64_t local_count  = disk::applyEnergyFloor(first, last, d);
            uint64_t global_count = 0;
            MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            if (Base::rank_ == 0)
                std::printf("Energy floor: %llu particles clamped to u_inf/10\n",
                            static_cast<unsigned long long>(global_count));
        }
        timer.step("applyEnergyFloor");

        /*FR:
        Tally mass/energy of particles about to be removed for failing neighbor-search convergence,
        strictly before updateSmoothingLength() removes them (same nc[] snapshot).
        Not attributable to either star (this failure isn't tied to proximity to a sink).
        */
        {
            auto [mass_nc, energy_nc] = disk::tallyNonConvergedRemoval(first, last, d, star1, star2);
            double local_vals[2]  = {mass_nc, energy_nc};
            double global_vals[2] = {0., 0.};
            MPI_Allreduce(local_vals, global_vals, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            nonconvergedRemovedMassAccum_   += global_vals[0];
            nonconvergedRemovedEnergyAccum_ += global_vals[1];
        }

        //FR: updates smoothing length of each particle to reach target number of neighbors ng0
        bool haveUnconvergedParticles = updateSmoothingLength(Base::groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }
        timer.step("UpdateQuantities");

        /*FR:
            1) Collects gravitational reaction forces from all MPI ranks for both stars
            2) Computes star potentials and energies (before position update, so KE = v_{n+1/2}^2)
            3) Integrates both stars’ positions
        */
        disk::computeAndExchangeBinaryPositions(star1, star2, d);
        timer.step("computeAndExchangeBinaryPositions");

        /*FR:
        Determines which particles:
            1) Fall inside accretion radius -> accrete them to star
            2) Become numerically problematic -> remove them
        Also tracks accretion and removal statistics (mass, momentum) to ensure conservation properties
        
        Practically: removes particles and writes:
            star.accreted_local (stores accretion statistics at given timestep)
            star.removed_local (stores removal statistics at given timestep)

            DOES NOT MODIFY STAR'S MASS OR MOMENTUM YET (see below)
        */
        disk::computeBinaryAccretionCondition(first, last, d, star1, star2);
        timer.step("computeAccretionCondition");

        //disk::computeBinaryAccretionCondition(star1, star2); //TODO: check if stars have merged (?)

        /*FR:
            1) Collects accreted mass and momentum from all MPI ranks
            2) Adds mass and momentum to star
        */
        // Returns the star-independent limit_h/r/z removal tally, accumulated globally here.
        // Pass d.minDt (current step): this runs after the star position update, whose displacement
        // (star.position_m1) spans the current step, so the star velocity is position_m1/minDt.
        const auto limitRemoved = disk::exchangeAndAccreteBinaryStars(star1, star2, d.minDt, Base::rank_);
        limitRemovedMassAccum_ += limitRemoved.mass;
        limitRemovedEnergyAccum_ += limitRemoved.energy;
        timer.step("exchangeAndAccreteOnStar");

        if (Base::rank_ == 0)
        {
            std::printf("star 1 position: %lf\t%lf\t%lf\n", star1.position[0], star1.position[1], star1.position[2]);
            std::printf("star 1 mass: %lf\n", star1.m);
            std::printf("additional pot. erg. 1: %lf\n", star1.potential);

            std::printf("star 2 position: %lf\t%lf\t%lf\n", star2.position[0], star2.position[1], star2.position[2]);
            std::printf("star 2 mass: %lf\n", star2.m);
            std::printf("additional pot. erg. 2: %lf\n", star2.potential);
        }

    }

    void printIterationTimings(const DomainType& domain, const DataType& simData) override
    {
        if (Base::rank_ == 0)
        {
            const auto& d = simData.hydro;
            this->out << "### Check ### Star 1 energy: " << star1.etot
                      << ", (kinetic: " << star1.ecin << ", potential: " << star1.egrav << ")\n";
            this->out << "### Check ### Star 2 energy: " << star2.etot
                      << ", (kinetic: " << star2.ecin << ", potential: " << star2.egrav << ")\n";
            this->out << "### Check ### Total energy (disk + stars): "
                      << d.etot + star1.etot + star2.etot << "\n";
        }
        Base::printIterationTimings(domain, simData);
    }

};

} // namespace sphexa