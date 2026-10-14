package com.rohit.georeckon.data.model

/**
 * Navigation positioning operational modes as per GeoReckon revised architecture.
 */
enum class PositioningMode {
    INITIALIZING,
    GNSS_AIDED,
    DEAD_RECKONING,
    REACQUIRING
}
