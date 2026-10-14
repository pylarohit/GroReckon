package com.rohit.georeckon.data.model

/**
 * Health assessment of the sensor & positioning estimator pipeline.
 */
enum class EngineHealth {
    NORMAL,
    ADAPTIVE,
    DEGRADED,
    RECOVERING
}
