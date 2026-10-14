package com.rohit.georeckon.engine

import com.rohit.georeckon.data.model.GnssSample
import com.rohit.georeckon.data.model.ImuSample
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Phase 4: Invariant Error-State Extended Kalman Filter (InEKF)
 * Handles IMU Dead Reckoning propagation, NHC constraints, ZUPT, and Mahalanobis gated GNSS fusion.
 */
class InvariantEKF {

    // Filter State
    var localX: Double = 0.0
        private set
    var localY: Double = 0.0
        private set
    var localZ: Double = 0.0
        private set
    var velX: Double = 0.0
        private set
    var velY: Double = 0.0
        private set
    var velZ: Double = 0.0
        private set
    var headingRad: Double = 0.0
        private set

    // Biases
    var accelBiasX: Double = 0.0
    var accelBiasY: Double = 0.0
    var accelBiasZ: Double = 0.0
    var gyroBiasZ: Double = 0.0

    // Covariances [P_pos_x, P_pos_y, P_vel_x, P_vel_y, P_head, P_bias]
    val covariance = DoubleArray(6) { 0.1 }

    // Reference Origin (Lat/Lng to local ENU conversion)
    var refLat: Double? = null
        private set
    var refLng: Double? = null
        private set

    private var lastTimestampNs: Long = 0L

    /**
     * Initializes filter origin with known GNSS fix.
     */
    fun initializeOrigin(lat: Double, lng: Double, heading: Double = 0.0) {
        refLat = lat
        refLng = lng
        headingRad = heading
        localX = 0.0
        localY = 0.0
        localZ = 0.0
        velX = 0.0
        velY = 0.0
        velZ = 0.0
        covariance.fill(0.05)
    }

    /**
     * IMU State Propagation (called on every IMU event).
     */
    fun propagate(imu: ImuSample, vehicleVec: DoubleArray) {
        if (lastTimestampNs == 0L) {
            lastTimestampNs = imu.timestampNs
            return
        }

        val dt = (imu.timestampNs - lastTimestampNs) / 1e9
        lastTimestampNs = imu.timestampNs

        if (dt <= 0 || dt > 1.0) return

        // 1. Gyroscope Heading Integration
        val unscaledGyroZ = imu.gyroZ - gyroBiasZ
        headingRad += unscaledGyroZ * dt
        headingRad = (headingRad + Math.PI * 2) % (Math.PI * 2)

        // 2. Acceleration in World Frame (ENU navigation frame: 0=North, 90=East)
        val fwdAcc = vehicleVec[0] - accelBiasX
        val latAcc = vehicleVec[1] - accelBiasY

        val worldAccX = fwdAcc * sin(headingRad) + latAcc * cos(headingRad) // East (X)
        val worldAccY = fwdAcc * cos(headingRad) - latAcc * sin(headingRad) // North (Y)

        // 3. Integration: v_next = v + a*dt, p_next = p + v*dt + 0.5*a*dt²
        localX += velX * dt + 0.5 * worldAccX * dt * dt
        localY += velY * dt + 0.5 * worldAccY * dt * dt

        velX += worldAccX * dt
        velY += worldAccY * dt

        // 4. Covariance Growth during Dead Reckoning
        covariance[0] += 0.08 * dt // Pos X var
        covariance[1] += 0.08 * dt // Pos Y var
        covariance[2] += 0.05 * dt // Vel X var
        covariance[3] += 0.05 * dt // Vel Y var
        covariance[4] += 0.01 * dt // Heading var
    }

    /**
     * Fuse AI Speed & Variance prediction (KinoNet-R2 measurement update).
     */
    fun fuseAiSpeed(predictedSpeedMps: Double, aiVariance: Double) {
        val currentSpeed = sqrt(velX * velX + velY * velY)
        val innovation = predictedSpeedMps - currentSpeed

        // Measurement noise R
        val r = aiVariance.coerceIn(0.01, 5.0)
        val k = covariance[2] / (covariance[2] + r) // Kalman Gain for velocity

        val correctedSpeed = currentSpeed + k * innovation
        velX = correctedSpeed * sin(headingRad)
        velY = correctedSpeed * cos(headingRad)

        covariance[2] *= (1.0 - k)
        covariance[3] *= (1.0 - k)
    }

    /**
     * Apply Non-Holonomic Constraint (NHC: lateral velocity ≈ 0).
     */
    fun applyNHC() {
        val fwdVel = velX * sin(headingRad) + velY * cos(headingRad)
        velX = fwdVel * sin(headingRad)
        velY = fwdVel * cos(headingRad)
        covariance[3] *= 0.9
    }

    /**
     * Apply Zero Velocity Update (ZUPT: velocity ≈ 0 when stationary).
     */
    fun applyZUPT() {
        velX = 0.0
        velY = 0.0
        velZ = 0.0
        covariance[2] = 0.01
        covariance[3] = 0.01
    }

    /**
     * Fuses valid GNSS location with Mahalanobis Innovation Gate check.
     */
    fun fuseGnss(gnss: GnssSample): Boolean {
        val originLat = refLat ?: run {
            initializeOrigin(gnss.latitude, gnss.longitude)
            return true
        }
        val originLng = refLng ?: return false

        // Convert GNSS Lat/Lng to local ENU meters
        val (gnssX, gnssY) = latLngToEnu(gnss.latitude, gnss.longitude, originLat, originLng)

        val innX = gnssX - localX
        val innY = gnssY - localY
        val mahalanobisDist = (innX * innX) / covariance[0] + (innY * innY) / covariance[1]

        // Mahalanobis Innovation Gating Check (adaptive threshold during recovery)
        val isRecovering = covariance[0] > 5.0 || covariance[1] > 5.0
        val gateThreshold = if (isRecovering) 100.0 else 25.0

        if (mahalanobisDist > gateThreshold && (gnss.accuracyM ?: 50.0) > 30.0) {
            return false
        }

        // Kalman update with adaptive gain for fast re-convergence on recovery
        val kPos = if (isRecovering) 0.95 else 0.85
        localX += kPos * innX
        localY += kPos * innY

        gnss.speedMps?.let { spd ->
            velX = spd * cos(headingRad)
            velY = spd * sin(headingRad)
        }

        // Reset uncertainty after GNSS fusion
        covariance[0] = (gnss.accuracyM ?: 5.0) * 0.2
        covariance[1] = (gnss.accuracyM ?: 5.0) * 0.2
        covariance[2] = 0.1
        covariance[3] = 0.1

        return true
    }

    /**
     * Returns current local ENU position converted back to Lat/Lng coordinates.
     */
    fun getEstimatedLatLng(): Pair<Double, Double>? {
        val oLat = refLat ?: return null
        val oLng = refLng ?: return null
        return enuToLatLng(localX, localY, oLat, oLng)
    }

    companion object {
        private const val EARTH_RADIUS_M = 6378137.0

        fun latLngToEnu(lat: Double, lng: Double, refLat: Double, refLng: Double): Pair<Double, Double> {
            val dLat = Math.toRadians(lat - refLat)
            val dLng = Math.toRadians(lng - refLng)
            val x = dLng * cos(Math.toRadians(refLat)) * EARTH_RADIUS_M
            val y = dLat * EARTH_RADIUS_M
            return Pair(x, y)
        }

        fun enuToLatLng(x: Double, y: Double, refLat: Double, refLng: Double): Pair<Double, Double> {
            val dLat = Math.toDegrees(y / EARTH_RADIUS_M)
            val dLng = Math.toDegrees(x / (EARTH_RADIUS_M * cos(Math.toRadians(refLat))))
            return Pair(refLat + dLat, refLng + dLng)
        }
    }
}
