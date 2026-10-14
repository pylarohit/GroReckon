package com.rohit.georeckon.engine

import com.rohit.georeckon.data.model.ImuSample
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Phase 2: Phone-to-Vehicle Coordinate Frame Alignment
 * Estimates gravity vector (pitch/roll) during stationary periods and forward direction upon acceleration.
 */
class FrameAlignment {

    var pitchRad: Double = 0.0
        private set
    var rollRad: Double = 0.0
        private set
    var yawOffsetRad: Double = 0.0
        private set

    var isAligned: Boolean = false
        private set
    var alignmentQuality: Double = 0.0
        private set

    private val stationaryBufferAcc = mutableListOf<DoubleArray>()
    private val motionBufferAcc = mutableListOf<DoubleArray>()
    private val requiredStationarySamples = 30

    /**
     * Updates alignment estimation using incoming filtered IMU sample.
     */
    fun update(sample: ImuSample): Boolean {
        val mag = sample.accelMagnitude
        val isStationary = abs(mag - 9.81) < 0.4 && sample.gyroMagnitude < 0.15

        if (isStationary) {
            stationaryBufferAcc.add(doubleArrayOf(sample.accelX, sample.accelY, sample.accelZ))
            if (stationaryBufferAcc.size >= requiredStationarySamples) {
                estimateTiltFromGravity()
                isAligned = true
                alignmentQuality = 0.95
            }
        } else if (isAligned) {
            motionBufferAcc.add(doubleArrayOf(sample.accelX, sample.accelY, sample.accelZ))
            if (motionBufferAcc.size in 10..60) {
                estimateForwardAxis()
            }
        }

        return isAligned
    }

    private fun estimateTiltFromGravity() {
        var meanX = 0.0; var meanY = 0.0; var meanZ = 0.0
        val count = stationaryBufferAcc.size
        for (a in stationaryBufferAcc) {
            meanX += a[0]; meanY += a[1]; meanZ += a[2]
        }
        meanX /= count; meanY /= count; meanZ /= count

        rollRad = atan2(meanY, meanZ)
        pitchRad = atan2(-meanX, sqrt(meanY * meanY + meanZ * meanZ))
    }

    private fun estimateForwardAxis() {
        if (motionBufferAcc.isEmpty()) return
        var meanX = 0.0; var meanY = 0.0
        for (a in motionBufferAcc) {
            meanX += a[0]; meanY += a[1]
        }
        yawOffsetRad = atan2(meanY, meanX)
    }

    /**
     * Transforms 3D vector [x, y, z] from Phone IMU frame to Vehicle Frame.
     */
    fun transformToVehicleFrame(vec: DoubleArray): DoubleArray {
        val x = vec[0]; val y = vec[1]; val z = vec[2]

        // Pitch & Roll rotation
        val cosR = cos(rollRad); val sinR = sin(rollRad)
        val cosP = cos(pitchRad); val sinP = sin(pitchRad)

        val y1 = y * cosR - z * sinR
        val z1 = y * sinR + z * cosR

        val x2 = x * cosP + z1 * sinP
        val z2 = -x * sinP + z1 * cosP

        // Yaw alignment
        val cosY = cos(yawOffsetRad); val sinY = sin(yawOffsetRad)
        val vFwd = x2 * cosY + y1 * sinY
        val vLat = -x2 * sinY + y1 * cosY

        return doubleArrayOf(vFwd, vLat, z2)
    }
}
