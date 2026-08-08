//
// Created by Noah Kubli on 07.03.2024.
//

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>

#include "central_potential.hpp"
#include "cstone/tree/definitions.h"
#include "removal_statistics.hpp"

namespace disk
{

struct StarData
{
    //! @brief The type of the potential to use when computing the gravitational forces involving the central star
    StarPotentialType potentialType{StarPotentialType::newtonian};

    //! @brief position of the central star
    cstone::Vec3<double> position{};

    //! @brief displacement of the central star in the last step
    cstone::Vec3<double> position_m1{};

    //! @brief mass of the central star
    double m{1e6};

    //! @brief inner size of the central star where particles are accreted
    double inner_size{0.};

    //! @brief Gravitational softening length for star-particle interactions.
    double grav_softening{0.};

    //! @brief Fix the position of the central star instead of integrating the position
    int fixed_star{1};

    //! @brief Reference value of beta at r = r_beta0: beta(r) = beta * (r / r_beta0)^(-beta_b)
    double beta{std::numeric_limits<double>::infinity()};

    //! @brief Power-law index for radially dependent beta; 0 = spatially constant (default)
    double beta_b{0.};

    //! @brief Reference radius for the beta power law; irrelevant when beta_b == 0
    double r_beta0{1.};

    //! @brief Remove all particles with a smoothing length greater than this value
    double removal_limit_h{std::numeric_limits<double>::infinity()};

    //! @brief Remove all particles beyond this cylindrical radius (sqrt(x²+y²) > removal_limit_r)
    double removal_limit_r{std::numeric_limits<double>::infinity()};

    //! @brief Remove all particles beyond this vertical height (|z| > removal_limit_z)
    double removal_limit_z{std::numeric_limits<double>::infinity()};

    //! @brief Don't cool any particle above this density threshold
    double cooling_rho_limit{std::numeric_limits<float>::infinity()};

    //! @brief Don't cool any particle whose internal energy is below
    double u_floor{0.};

    //! @brief Limit the timestep depending on changes in the internal energy. delta_t = K_u * u / du
    double K_u{std::numeric_limits<double>::infinity()};

    //! @brief Softening length for beta cooling to prevent du divergence near the center of mass
    double betaEps{0.1};

    //! @brief Characteristic cavity size L used to blend the beta-cooling center between the
    //!        primary star (binary separation d >> L) and the binary center of mass (d << L).
    //!        Only star1's value is used (see betaCoolingCenter).
    double beta_cooling_L{1.};

    /*FR:
    Reads or writes star attributes depending on what Archive object is passed. 
        - Used during initialization to define star object (called in propagator->load() )
        - Used during particle dumps to write to file (called in propagator->save() )
    NOTE: this does not initialize the star's initial position!
    */
    template<typename Archive>
    void loadOrStoreAttributes(Archive* ar, const std::string& prefix = "star")
    {
        /*FR
        Loads or stores an attribute of the star:
            - attribute: name of the attribute to load/store (in the Archive)
            - location: memory location of the attribute to store (in the star)
            - attrSize: size of the attribute
        */
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                if constexpr (std::is_enum_v<std::decay_t<decltype(*location)>>)
                {
                    // handle pointers to enum by casting to the underlying type
                    using EType = std::decay_t<decltype(*location)>;
                    using UType = std::underlying_type_t<EType>;
                    auto tmp    = static_cast<UType>(*location);
                    ar->stepAttribute(attribute, &tmp, attrSize);
                    *location = static_cast<EType>(tmp);
                }
                else { ar->stepAttribute(attribute, location, attrSize); }
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute " << attribute
                              << " not set in file or initializer, setting to default value " << *location << std::endl;
                }
            }
        };

        /*! @brief Load or store a per-star hyperparameter -- configuration fixed for the whole run.
         *
         *  Stored once in the file root instead of being repeated in every step group. On read the root
         *  wins but the step group is still tried, so existing initial conditions (which write these
         *  per-step) and files from before this split keep working untouched.
         */
        auto constantIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            using LocType = std::decay_t<decltype(*location)>;

            auto io = [ar, &attribute, attrSize](auto* loc)
            {
                // In-memory archives (Builtin*) have no root/step distinction; leave them as-is.
                if constexpr (requires { ar->fileAttribute(attribute, loc, attrSize); })
                {
                    if constexpr (requires { ar->fileAttributes(); })   // reader: root, else fall back to step
                    {
                        // Fall back to the step group on any failure, not just absence: pre-split files
                        // kept the correctly-typed value per step and a loosely-typed (float64) copy at
                        // the root, so a root read there fails with a type mismatch, not out_of_range.
                        try { ar->fileAttribute(attribute, loc, attrSize); }
                        catch (std::exception&) { ar->stepAttribute(attribute, loc, attrSize); }
                    }
                    else { ar->fileAttribute(attribute, loc, attrSize); }
                }
                else { ar->stepAttribute(attribute, loc, attrSize); }
            };

            try
            {
                if constexpr (std::is_enum_v<LocType>)
                {
                    using UType = std::underlying_type_t<LocType>;
                    auto tmp    = static_cast<UType>(*location);
                    io(&tmp);
                    *location = static_cast<LocType>(tmp);
                }
                else { io(location); }
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute " << attribute
                              << " not set in file or initializer, setting to default value " << *location << std::endl;
                }
            }
        };

        // ---- time-dependent star state: one value per step ----
        optionalIO(prefix + "::x", &position[0], 1);
        optionalIO(prefix + "::y", &position[1], 1);
        optionalIO(prefix + "::z", &position[2], 1);
        optionalIO(prefix + "::x_m1", &position_m1[0], 1);
        optionalIO(prefix + "::y_m1", &position_m1[1], 1);
        optionalIO(prefix + "::z_m1", &position_m1[2], 1);
        optionalIO(prefix + "::m", &m, 1);              // grows by accretion
        optionalIO(prefix + "::spin_x", &spin[0], 1);
        optionalIO(prefix + "::spin_y", &spin[1], 1);
        optionalIO(prefix + "::spin_z", &spin[2], 1);

        // ---- per-star hyperparameters: written once at the file root ----
        constantIO(prefix + "::potentialType", &potentialType, 1);
        constantIO(prefix + "::inner_size", &inner_size, 1);
        constantIO(prefix + "::grav_softening", &grav_softening, 1);
        constantIO(prefix + "::fixed_star", &fixed_star, 1);
        constantIO(prefix + "::beta", &beta, 1);
        constantIO(prefix + "::beta_b", &beta_b, 1);
        constantIO(prefix + "::r_beta0", &r_beta0, 1);
        constantIO(prefix + "::removal_limit_h", &removal_limit_h, 1);
        constantIO(prefix + "::removal_limit_r", &removal_limit_r, 1);
        constantIO(prefix + "::removal_limit_z", &removal_limit_z, 1);
        constantIO(prefix + "::cooling_rho_limit", &cooling_rho_limit, 1);
        constantIO(prefix + "::u_floor", &u_floor, 1);
        constantIO(prefix + "::K_u", &K_u, 1);
        constantIO(prefix + "::betaEps", &betaEps, 1);
        constantIO(prefix + "::beta_cooling_L", &beta_cooling_L, 1);

        // Instantaneous diagnostics (safe to read back on restart)
        optionalIO(prefix + "::acc_x", &acc[0], 1);
        optionalIO(prefix + "::acc_y", &acc[1], 1);
        optionalIO(prefix + "::acc_z", &acc[2], 1);
        optionalIO(prefix + "::ecin", &ecin, 1);
        optionalIO(prefix + "::egrav", &egrav, 1);
        optionalIO(prefix + "::etot", &etot, 1);

        // Diagnostics accumulated since the last dump. NOTE: these must be reset to zero after save()
        // (BinaryProp::save) and after load() (BinaryProp::load) -- see callers -- so that a restart
        // never resumes mid-window with stale totals read back from the file.
        optionalIO(prefix + "::accreted_mass_accum", &accreted_mass_accum, 1);
        optionalIO(prefix + "::accreted_count_accum", &accreted_count_accum, 1);
        optionalIO(prefix + "::accreted_ke_accum", &accreted_ke_accum, 1);
        optionalIO(prefix + "::accreted_energy_accum", &accreted_energy_accum, 1);
        optionalIO(prefix + "::accreted_Lx_accum", &accreted_L_accum[0], 1);
        optionalIO(prefix + "::accreted_Ly_accum", &accreted_L_accum[1], 1);
        optionalIO(prefix + "::accreted_Lz_accum", &accreted_L_accum[2], 1);
        optionalIO(prefix + "::accreted_px_accum", &accreted_p_accum[0], 1);
        optionalIO(prefix + "::accreted_py_accum", &accreted_p_accum[1], 1);
        optionalIO(prefix + "::accreted_pz_accum", &accreted_p_accum[2], 1);
    };

    //! @brief Reset all diagnostics accumulated since the last dump to zero. Must be called right after
    //!        writing a dump (BinaryProp::save) and right after loading initial conditions/a restart
    //!        (BinaryProp::load), so a fresh window always starts clean.
    void resetAccumulatedDiagnostics()
    {
        accreted_mass_accum    = 0.;
        accreted_count_accum   = 0;
        accreted_ke_accum      = 0.;
        accreted_energy_accum  = 0.;
        accreted_L_accum       = {0., 0., 0.};
        accreted_p_accum       = {0., 0., 0.};
    }

    //! @brief Specific potential at star location due to disk particles: ∑_i (-G m_i / r_i).
    //!        Units are [L²/T²] (energy per unit mass). Disk-star interaction energy = star.m * star.potential.
    double potential{};

    double ecin{0.};   //! @brief kinetic energy of the star
    double egrav{0.};  //! @brief gravitational PE: star.m * star.potential + 0.5 * star-star PE
    double etot{0.};   //! @brief total mechanical energy = ecin + egrav

    //! @brief Specific binary potential [0] (= -G*M_other/r, units [L^2/T^2]) and
    //!        total binary force [1..3] ([Force]) acting on the star due to the other star.
    cstone::Vec4<double> force_binary{0, 0, 0, 0};

    //! @brief Values local to each rank
    //! @brief Specific potential [0] and force [1..3] acting on the star due to particles (local to rank)
    cstone::Vec4<double> force_local{};

    //! @brief Statistics of accreted particles
    RemovalStatistics accreted_local;

    //! @brief Statistics of removed particles
    RemovalStatistics removed_local;

    //! @brief Instantaneous star acceleration (disk + companion gravity), captured every step.
    cstone::Vec3<double> acc{0., 0., 0.};

    //! @brief Diagnostics accumulated since the last dump. All zeroed by resetAccumulatedDiagnostics()
    //!        right after a dump is written and right after load()/restart.
    double accreted_mass_accum{0.};
    uint64_t accreted_count_accum{0};
    double accreted_ke_accum{0.};
    //! @brief Sum of each accreted particle's own pre-accretion total energy (KE + u + potential).
    double accreted_energy_accum{0.};
    //! @brief Delta of the star's own angular momentum (about the fixed coordinate origin) due to accretion.
    cstone::Vec3<double> accreted_L_accum{0., 0., 0.};
    //! @brief Delta of the star's own linear momentum due to accretion.
    cstone::Vec3<double> accreted_p_accum{0., 0., 0.};

    /*! @note There is deliberately no per-star removal accumulator here. The limit_h/r/z cutoffs are
     *  properties of the particle relative to the origin and never reference a star position, so removals
     *  are star-independent and are accumulated globally by BinaryProp instead. Tallying them per-star is
     *  what previously produced bit-identical star1::/star2::removed_*_accum in every dump.
     */

    //! @brief Spin angular momentum of the star, accumulated by accretion (eq 154).
    cstone::Vec3<double> spin{0., 0., 0.};

    //! @brief timestep from central acceleration (local to rank)
    double t_star{};

    //! @brief timestep from binary acceleration
    double t_binary{0};

    //! @brief du-timestep (local to rank)
    double t_du{};
};

//! @brief Compute the beta-cooling center and effective mass as a smooth blend between the
//!        primary star and the binary center of mass, controlled by the binary separation d.
//!
//!        A shifted logistic of width 0.1*L, centered at d = L (L = star1.beta_cooling_L, the
//!        characteristic cavity size), sets the blend weight:
//!            sigma  = 1 / (1 + exp(-(d - L) / (0.1 L)))
//!            center = sigma * x_primary + (1 - sigma) * x_com
//!            M_eff  = sigma * m_primary + (1 - sigma) * M_total
//!
//!        d >> L (secondary far outside, e.g. during relaxation): sigma -> 1, so cooling is
//!        centered on the primary with mass m1. d << L (compact binary after cavity formation):
//!        sigma -> 0, so cooling is centered on the binary COM with the total mass. star1 is the
//!        primary (placed at the system center); star2 is the secondary.
template<typename StarData>
void betaCoolingCenter(const StarData& star1, const StarData& star2,
                       cstone::Vec3<double>& center, double& M_eff)
{
    const double M_total = star1.m + star2.m;
    const cstone::Vec3<double> com{
        (star1.m * star1.position[0] + star2.m * star2.position[0]) / M_total,
        (star1.m * star1.position[1] + star2.m * star2.position[1]) / M_total,
        (star1.m * star1.position[2] + star2.m * star2.position[2]) / M_total};

    const double dx    = star2.position[0] - star1.position[0];
    const double dy    = star2.position[1] - star1.position[1];
    const double dz    = star2.position[2] - star1.position[2];
    const double d     = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double L     = star1.beta_cooling_L;
    const double sigma = 1.0 / (1.0 + std::exp(-(d - L) / (0.1 * L)));

    center[0] = sigma * star1.position[0] + (1.0 - sigma) * com[0];
    center[1] = sigma * star1.position[1] + (1.0 - sigma) * com[1];
    center[2] = sigma * star1.position[2] + (1.0 - sigma) * com[2];
    M_eff     = sigma * star1.m + (1.0 - sigma) * M_total;
}

} // namespace disk
