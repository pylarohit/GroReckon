package com.rohit.georeckon.sensors

import android.annotation.SuppressLint
import android.content.Context
import android.os.Looper
import com.google.android.gms.location.*
import com.rohit.georeckon.data.model.GnssSample

/**
 * Manages live Android GNSS / FusedLocationProvider fixes.
 * Detects GPS signal loss and recovery to trigger dead-reckoning handoff.
 */
class GnssManager(context: Context) {

    private val fusedLocationClient = LocationServices.getFusedLocationProviderClient(context)
    private var locationCallback: LocationCallback? = null

    var isListening = false
        private set

    /** True when the last LocationAvailability said GPS is available */
    var isGpsAvailable = false
        private set

    /** Set these to receive GPS loss / regain events */
    var onGpsLost: (() -> Unit)? = null
    var onGpsRegained: (() -> Unit)? = null

    /** Minimum acceptable accuracy (metres) – fixes worse than this are discarded */
    private val minAccuracyM = 100.0

    @SuppressLint("MissingPermission")
    fun startListening(onGnssSample: (GnssSample) -> Unit) {
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateIntervalMillis(500L)
            .setWaitForAccurateLocation(false)
            .build()

        locationCallback = object : LocationCallback() {

            override fun onLocationAvailability(availability: LocationAvailability) {
                val nowAvailable = availability.isLocationAvailable
                if (isGpsAvailable && !nowAvailable) {
                    // GPS just lost
                    isGpsAvailable = false
                    onGpsLost?.invoke()
                } else if (!isGpsAvailable && nowAvailable) {
                    // GPS just regained
                    isGpsAvailable = true
                    onGpsRegained?.invoke()
                }
            }

            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return

                // Mark GPS available on any valid fix
                if (!isGpsAvailable) {
                    isGpsAvailable = true
                    onGpsRegained?.invoke()
                }

                // Discard fixes that are too inaccurate (e.g. network-only in a tunnel)
                val accuracyM = if (loc.hasAccuracy()) loc.accuracy.toDouble() else 99.0
                val isValid = accuracyM <= minAccuracyM

                val sample = GnssSample(
                    timestampNs = loc.elapsedRealtimeNanos,
                    latitude = loc.latitude,
                    longitude = loc.longitude,
                    speedMps = if (loc.hasSpeed()) loc.speed.toDouble() else null,
                    accuracyM = accuracyM,
                    valid = isValid
                )
                onGnssSample(sample)
            }
        }

        fusedLocationClient.requestLocationUpdates(req, locationCallback!!, Looper.getMainLooper())
        isListening = true
    }

    fun stopListening() {
        locationCallback?.let { fusedLocationClient.removeLocationUpdates(it) }
        isListening = false
        isGpsAvailable = false
    }
}
