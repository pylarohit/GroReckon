package com.rohit.georeckon.engine

import com.rohit.georeckon.data.model.ImuSample
import kotlin.math.abs

/**
 * Phase 1: Signal Cleaning & Spike Filtering
 * Implements a causal Butterworth 3Hz low-pass filter + 3x MAD online outlier rejection.
 */
class SignalFilter(
    private val sampleRateHz: Double = 50.0,
    private val cutoffFreqHz: Double = 3.0
) {
    // 2nd-order Butterworth low-pass coefficients (b0, b1, b2, a1, a2)
    private var b0: Double = 0.0
    private var b1: Double = 0.0
    private var b2: Double = 0.0
    private var a1: Double = 0.0
    private var a2: Double = 0.0

    // State buffers for 6 channels [ax, ay, az, gx, gy, gz]
    private val x1 = DoubleArray(6)
    private val x2 = DoubleArray(6)
    private val y1 = DoubleArray(6)
    private val y2 = DoubleArray(6)

    // Moving window for MAD (Median Absolute Deviation) spike cleaning
    private val historyAccX = ArrayDeque<Double>()
    private val historyAccY = ArrayDeque<Double>()
    private val historyAccZ = ArrayDeque<Double>()
    private val maxWindowSize = 15

    init {
        computeButterworthCoefficients()
    }

    private fun computeButterworthCoefficients() {
        val omega = 2.0 * Math.PI * cutoffFreqHz / sampleRateHz
        val sinW = Math.sin(omega)
        val cosW = Math.cos(omega)
        val alpha = sinW / (2.0 * Math.sqrt(2.0))

        val a0 = 1.0 + alpha
        b0 = ((1.0 - cosW) / 2.0) / a0
        b1 = (1.0 - cosW) / a0
        b2 = ((1.0 - cosW) / 2.0) / a0
        a1 = (-2.0 * cosW) / a0
        a2 = (1.0 - alpha) / a0
    }

    /**
     * Filters incoming raw IMU sample.
     * Applies 3x MAD spike clamping followed by 2nd-order IIR Butterworth filtering.
     */
    fun process(raw: ImuSample): ImuSample {
        // 1. Spike detection using MAD on Accel
        val cleanAx = clampSpike(raw.accelX, historyAccX)
        val cleanAy = clampSpike(raw.accelY, historyAccY)
        val cleanAz = clampSpike(raw.accelZ, historyAccZ)

        var quality = raw.qualityFlags
        if (cleanAx != raw.accelX || cleanAy != raw.accelY || cleanAz != raw.accelZ) {
            quality = quality or ImuSample.QUALITY_SPIKE_DETECTED
        }

        // 2. Butterworth IIR filtering for 6 channels
        val rawVector = doubleArrayOf(cleanAx, cleanAy, cleanAz, raw.gyroX, raw.gyroY, raw.gyroZ)
        val filteredVector = DoubleArray(6)

        for (i in 0 until 6) {
            val x = rawVector[i]
            val y = b0 * x + b1 * x1[i] + b2 * x2[i] - a1 * y1[i] - a2 * y2[i]

            // Shift states
            x2[i] = x1[i]
            x1[i] = x
            y2[i] = y1[i]
            y1[i] = y

            filteredVector[i] = y
        }

        return ImuSample(
            timestampNs = raw.timestampNs,
            accelX = filteredVector[0],
            accelY = filteredVector[1],
            accelZ = filteredVector[2],
            gyroX = filteredVector[3],
            gyroY = filteredVector[4],
            gyroZ = filteredVector[5],
            frameId = raw.frameId,
            qualityFlags = quality
        )
    }

    private fun clampSpike(valIn: Double, history: ArrayDeque<Double>): Double {
        history.addLast(valIn)
        if (history.size > maxWindowSize) history.removeFirst()

        if (history.size < 5) return valIn

        val sorted = history.sorted()
        val median = sorted[sorted.size / 2]
        val mad = sorted.map { abs(it - median) }.sorted().run { this[this.size / 2] }

        val threshold = 3.0 * (if (mad < 1e-4) 0.5 else mad)
        val diff = abs(valIn - median)

        return if (diff > threshold) {
            median + Math.signum(valIn - median) * threshold
        } else {
            valIn
        }
    }
}
