package com.rohit.georeckon.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "trips")
data class TripEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val startTimeMs: Long,
    val endTimeMs: Long,
    val mode: String,
    val modelVersion: String,
    val outageDurationSec: Int,
    val endpointErrorM: Double,
    val rmsPositionErrorM: Double,
    val maxErrorM: Double,
    val endpointDriftPercent: Double,
    val aiSpeedErrorMps: Double,
    val isDemo: Boolean = true
)
