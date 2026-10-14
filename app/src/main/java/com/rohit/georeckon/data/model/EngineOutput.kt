package com.rohit.georeckon.data.model

/**
 * Output packet emitted by the GeoReckon Navigation Engine at ~10 Hz.
 */
data class EngineOutput(
    val timestampNs: Long,
    val latitude: Double?,
    val longitude: Double?,
    val localX: Double,
    val localY: Double,
    val localZ: Double,
    val velocityX: Double,
    val velocityY: Double,
    val velocityZ: Double,
    val headingRad: Double,
    val covariance: DoubleArray = DoubleArray(6) { 0.1 },
    val positioningMode: PositioningMode,
    val health: EngineHealth,
    val alignmentHealthy: Boolean = true,
    val gnssAccepted: Boolean = true,
    val aiAccepted: Boolean = true,
    val modelVersion: String = "kinonet-r2-tflite-v1",
    val speedMps: Double = Math.sqrt(velocityX * velocityX + velocityY * velocityY),
    val uncertaintyM: Double = Math.sqrt(covariance.getOrElse(0) { 1.0 })
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as EngineOutput
        return timestampNs == other.timestampNs &&
                latitude == other.latitude &&
                longitude == other.longitude &&
                positioningMode == other.positioningMode
    }

    override fun hashCode(): Int {
        var result = timestampNs.hashCode()
        result = 31 * result + (latitude?.hashCode() ?: 0)
        result = 31 * result + (longitude?.hashCode() ?: 0)
        return result
    }
}
