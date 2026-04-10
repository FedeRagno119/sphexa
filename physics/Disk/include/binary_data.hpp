//
// Created by Federico Ragno on 11.03.2026.
//
#pragma once

namespace disk
{

template <typename StarData>
struct BinarySystem {

    StarData& star1 {};
    StarData& star2 {};
    
    /* 
    POSSIBLE ATTRIBUTES
    mass_ratio;
    eccentricity;
    center of mass;
    reduced mass;
    */

    /*
    FUNCTIONS TO IMPLEMENT:

        get_primary -> returns the primary star
        get_eccentricity
        get_separation
        ...
    
    */

};


template <typename StarData>
bool binary_is_consistent(const StarData& star1, const StarData& star2){
    
    const double dx = star2.position[0] - star1.position[0];
    const double dy = star2.position[1] - star1.position[1];
    const double dz = star2.position[2] - star1.position[2];
    const double dist  = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double min_dist = star1.inner_size + star2.inner_size;

    return (
        star1.potentialType == star2.potentialType  &&  // stars have consistent potential types
        dist > min_dist                             &&  // stars are not penetrating each other
        star1.t_binary == star2.t_binary                // stars have same binary orbital time
    );
}


}; // namespace disk