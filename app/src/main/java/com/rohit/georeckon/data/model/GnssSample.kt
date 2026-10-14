package com.rohit.georeckon.data.model

/**
 * GNSS location measurement sample.
 */
data class GnssSample(
    val timestampNs: Long,
    val latitude: Double,
    val longitude: Double,
    val speedMps: Double? = null,
    val accuracyM: Double? = null,
    val valid: Boolean = true
)
