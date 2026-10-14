package com.rohit.georeckon.engine

import android.content.Context
import com.rohit.georeckon.data.model.*
import kotlin.math.abs

/**
 * Phase 5: GeoReckon Master Navigation Engine
 * Orchestrates Phase 1 (Cleaning) → Phase 2 (Alignment) → Phase 3 (KinoNet-R2 AI) → Phase 4 (InEKF)
 */
class NavigationEngine(context: Context? = null) {

    val signalFilter = SignalFilter()
    val frameAlignment = FrameAlignment()
    val kinoNetRunner = KinoNetRunner(context)
    val ekf = InvariantEKF()

    var currentMode: PositioningMode = PositioningMode.INITIALIZING
        private set
    var currentHealth: EngineHealth = EngineHealth.NORMAL
        private set

    var isGnssOutageSimulated: Boolean = false

    // Telemetry stats
    var lastAiSpeedMps: Double = 0.0
        private set
    var lastAiVariance: Double = 0.05
        private set
    var lastAiAccepted: Boolean = true
        private set
    var lastGnssAccepted: Boolean = true
        private set

    private var outputListener: ((EngineOutput) -> Unit)? = null

    fun setOutputListener(listener: (EngineOutput) -> Unit) {
        outputListener = listener
    }

    /**
     * Ingests real or replayed raw IMU sample through the 5-phase pipeline.
     */
    fun processImu(rawSample: ImuSample): EngineOutput {
        // Phase 1: Signal Cleaning & 3x MAD Spike Filter
        val cleanImu = signalFilter.process(rawSample)

        // Phase 2: Phone-to-Vehicle Alignment
        val aligned = frameAlignment.update(cleanImu)
        val vehicleVec = frameAlignment.transformToVehicleFrame(
            doubleArrayOf(cleanImu.accelX, cleanImu.accelY, cleanImu.accelZ)
        )

        // Phase 3: KinoNet-R2 AI Speed & Variance Model
        val aiPred = kinoNetRunner.pushSample(cleanImu)
        if (aiPred != null) {
            lastAiSpeedMps = aiPred.speedMps
            lastAiVariance = aiPred.variance
            lastAiAccepted = aiPred.isAccepted
        }

        // Phase 4: Invariant EKF Propagation
        ekf.propagate(cleanImu, vehicleVec)

        // Apply AI speed measurement update if valid
        if (aiPred != null && aiPred.isAccepted) {
            ekf.fuseAiSpeed(aiPred.speedMps, aiPred.variance)
        }

        // Apply NHC & ZUPT constraints
        ekf.applyNHC()

        // Only apply ZUPT when vehicle is ACTUALLY stationary:
        // Must have near-zero AI forward speed AND stationary IMU energy
        val isZeroMotion = aiPred != null && aiPred.speedMps < 0.25
        val isStationarySensors = abs(cleanImu.accelMagnitude - 9.81) < 0.15 && cleanImu.gyroMagnitude < 0.05
        if (isZeroMotion && isStationarySensors) {
            ekf.applyZUPT()
        }

        // Determine Mode & Health State
        updateEngineState(cleanImu.timestampNs)

        val latLng = ekf.getEstimatedLatLng()

        val output = EngineOutput(
            timestampNs = cleanImu.timestampNs,
            latitude = latLng?.first,
            longitude = latLng?.second,
            localX = ekf.localX,
            localY = ekf.localY,
            localZ = ekf.localZ,
            velocityX = ekf.velX,
            velocityY = ekf.velY,
            velocityZ = ekf.velZ,
            headingRad = ekf.headingRad,
            covariance = ekf.covariance,
            positioningMode = currentMode,
            health = currentHealth,
            alignmentHealthy = aligned,
            gnssAccepted = lastGnssAccepted,
            aiAccepted = lastAiAccepted
        )

        outputListener?.invoke(output)
        return output
    }

    private var lastGnssTimestampNs: Long = 0L

    /**
     * Ingests GNSS measurement sample.
     */
    fun processGnss(sample: GnssSample): Boolean {
        if (isGnssOutageSimulated || !sample.valid) {
            lastGnssAccepted = false
            return false
        }

        lastGnssTimestampNs = sample.timestampNs
        val accepted = ekf.fuseGnss(sample)
        lastGnssAccepted = accepted

        if (accepted) {
            when (currentMode) {
                PositioningMode.DEAD_RECKONING -> {
                    currentMode = PositioningMode.REACQUIRING
                    currentHealth = EngineHealth.RECOVERING
                }
                PositioningMode.REACQUIRING -> {
                    currentMode = PositioningMode.GNSS_AIDED
                    currentHealth = EngineHealth.NORMAL
                }
                PositioningMode.INITIALIZING -> {
                    currentMode = PositioningMode.GNSS_AIDED
                    currentHealth = EngineHealth.NORMAL
                }
                PositioningMode.GNSS_AIDED -> {
                    currentHealth = EngineHealth.NORMAL
                }
            }
        }

        return accepted
    }

    private fun updateEngineState(currentTimestampNs: Long) {
        if (isGnssOutageSimulated) {
            currentMode = PositioningMode.DEAD_RECKONING
            currentHealth = EngineHealth.ADAPTIVE
        } else if (lastGnssTimestampNs > 0L && (currentTimestampNs - lastGnssTimestampNs) > 2_000_000_000L) {
            // More than 2 seconds without GNSS -> Dead Reckoning active
            currentMode = PositioningMode.DEAD_RECKONING
            currentHealth = EngineHealth.ADAPTIVE
        } else if (currentMode == PositioningMode.INITIALIZING && ekf.refLat != null) {
            currentMode = PositioningMode.GNSS_AIDED
            currentHealth = EngineHealth.NORMAL
        }
    }

    fun toggleGnssOutageSimulation(): Boolean {
        isGnssOutageSimulated = !isGnssOutageSimulated
        if (isGnssOutageSimulated) {
            currentMode = PositioningMode.DEAD_RECKONING
            currentHealth = EngineHealth.ADAPTIVE
        } else {
            currentMode = PositioningMode.REACQUIRING
            currentHealth = EngineHealth.RECOVERING
        }
        return isGnssOutageSimulated
    }
}
