package com.rohit.georeckon.data.replay

/**
 * Data model for offline demo trip replay JSON asset.
 */
data class DemoSamplePoint(
    val sec: Int,
    val lat: Double,
    val lng: Double,
    val ax: Double,
    val ay: Double,
    val az: Double,
    val gx: Double,
    val gy: Double,
    val gz: Double,
    val spd: Double,
    val gnss: Boolean
)

data class DemoTripMeta(
    val tripId: String,
    val sampleRateHz: Int,
    val outageStartSec: Int,
    val outageEndSec: Int,
    val modelVersion: String,
    val dataType: String,
    val startLat: Double,
    val startLng: Double,
    val samples: List<DemoSamplePoint>
)
