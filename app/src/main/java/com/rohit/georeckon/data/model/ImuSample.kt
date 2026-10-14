package com.rohit.georeckon.data.model

/**
 * Raw or preprocessed 6-axis IMU measurement sample.
 * Timestamps use actual hardware sensor time (nanoseconds).
 */
data class ImuSample(
    val timestampNs: Long,
    val accelX: Double,
    val accelY: Double,
    val accelZ: Double,
    val gyroX: Double,
    val gyroY: Double,
    val gyroZ: Double,
    val frameId: String = "phone_imu",
    val qualityFlags: Int = QUALITY_OK
) {
    companion object {
        const val QUALITY_OK = 0
        const val QUALITY_SPIKE_DETECTED = 1
        const val QUALITY_SATURATED = 2
        const val QUALITY_MISSING_SAMPLE = 4
    }

    /** Magnitude of total acceleration in m/s² */
    val accelMagnitude: Double
        get() = Math.sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ)

    /** Magnitude of total angular velocity in rad/s */
    val gyroMagnitude: Double
        get() = Math.sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ)
}
