//
// Created by Federico Ragno on 11.03.2026.
//
#pragma once

namespace disk
{

    /*TODO
    Needs to check for:
        1) stars have same potential type
        2) stars do not penetrate each other
        3) stars have same binary orbital time
        4) 
    */
    template <typename StarData>
    bool binary_is_consistent(const StarData& star1, const StarData& star2){
        return (
            star1.potentialType == star2.potentialType      // stars have consistent potential types
            && true
        );
    }

}