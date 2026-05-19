//
// Created by Noah Kubli on 07.03.2024.
//

#pragma once

#include <array>
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

    //! @brief Constant for beta cooling in the disk
    double beta{std::numeric_limits<double>::infinity()};

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

        optionalIO(prefix + "::potentialType", &potentialType, 1);
        optionalIO(prefix + "::x", &position[0], 1);
        optionalIO(prefix + "::y", &position[1], 1);
        optionalIO(prefix + "::z", &position[2], 1);
        optionalIO(prefix + "::x_m1", &position_m1[0], 1);
        optionalIO(prefix + "::y_m1", &position_m1[1], 1);
        optionalIO(prefix + "::z_m1", &position_m1[2], 1);
        optionalIO(prefix + "::m", &m, 1);
        optionalIO(prefix + "::inner_size", &inner_size, 1);
        optionalIO(prefix + "::grav_softening", &grav_softening, 1);
        optionalIO(prefix + "::fixed_star", &fixed_star, 1);
        optionalIO(prefix + "::beta", &beta, 1);
        optionalIO(prefix + "::removal_limit_h", &removal_limit_h, 1);
        optionalIO(prefix + "::removal_limit_r", &removal_limit_r, 1);
        optionalIO(prefix + "::removal_limit_z", &removal_limit_z, 1);
        optionalIO(prefix + "::cooling_rho_limit", &cooling_rho_limit, 1);
        optionalIO(prefix + "::u_floor", &u_floor, 1);
        optionalIO(prefix + "::K_u", &K_u, 1);
        optionalIO(prefix + "::betaEps", &betaEps, 1);
    };

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

    //! @brief timestep from central acceleration (local to rank)
    double t_star{};

    //! @brief timestep from binary acceleration
    double t_binary{0};

    //! @brief du-timestep (local to rank)
    double t_du{};
};
} // namespace disk
