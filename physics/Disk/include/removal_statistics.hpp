//
// Created by Noah Kubli on 06.12.2024.
//

#pragma once
#include "cstone/tree/definitions.h"

namespace disk
{
struct RemovalStatistics
{
    double               mass{0.};
    cstone::Vec3<double> momentum{0., 0., 0.};
    unsigned             count{0};
    cstone::Vec3<double> weighted_position{0., 0., 0.};  // Σ m_a * r_a  (eq 151)
    cstone::Vec3<double> angular_momentum{0., 0., 0.};   // Σ m_a * (r_a × v_a)  (eq 154)
    //! @brief Σ_a [ ½ m_a v_a² + m_a u_a + m_a ugrav_a + m_a·(star potential terms) ], each particle's own
    //!        total energy (kinetic + internal + potential) evaluated just before it is removed/accreted.
    double energy{0.};

    HOST_DEVICE_FUN friend RemovalStatistics operator+(const RemovalStatistics& a, const RemovalStatistics& b)
    {
        RemovalStatistics result;
        result.mass              = a.mass + b.mass;
        result.momentum          = a.momentum + b.momentum;
        result.count             = a.count + b.count;
        result.weighted_position = a.weighted_position + b.weighted_position;
        result.angular_momentum  = a.angular_momentum + b.angular_momentum;
        result.energy            = a.energy + b.energy;
        return result;
    }
};

} // namespace disk
