package com.rohit.georeckon.data.replay

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.rohit.georeckon.data.model.GnssSample
import com.rohit.georeckon.data.model.ImuSample
import com.rohit.georeckon.engine.NavigationEngine
import org.json.JSONObject

/**
 * Loads assets/demo/trip.json and plays back IMU + GNSS samples at 10Hz to drive NavigationEngine deterministically.
 */
class DemoTripRepository(private val context: Context) {

    private val handler = Handler(Looper.getMainLooper())
    private var playbackRunnable: Runnable? = null
    var isPlaying = false
        private set

    fun loadDemoTrip(): DemoTripMeta? {
        return try {
            val jsonStr = context.assets.open("demo/trip.json").bufferedReader().use { it.readText() }
            val obj = JSONObject(jsonStr)
            val arr = obj.getJSONArray("samples")
            val list = mutableListOf<DemoSamplePoint>()

            for (i in 0 until arr.length()) {
                val s = arr.getJSONObject(i)
                list.add(
                    DemoSamplePoint(
                        sec = s.getInt("sec"),
                        lat = s.getDouble("lat"),
                        lng = s.getDouble("lng"),
                        ax = s.getDouble("ax"),
                        ay = s.getDouble("ay"),
                        az = s.getDouble("az"),
                        gx = s.getDouble("gx"),
                        gy = s.getDouble("gy"),
                        gz = s.getDouble("gz"),
                        spd = s.getDouble("spd"),
                        gnss = s.getBoolean("gnss")
                    )
                )
            }

            DemoTripMeta(
                tripId = obj.getString("tripId"),
                sampleRateHz = obj.getInt("sampleRateHz"),
                outageStartSec = obj.getInt("outageStartSec"),
                outageEndSec = obj.getInt("outageEndSec"),
                modelVersion = obj.getString("modelVersion"),
                dataType = obj.getString("dataType"),
                startLat = obj.getDouble("startLat"),
                startLng = obj.getDouble("startLng"),
                samples = list
            )
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    fun startPlayback(
        engine: NavigationEngine,
        originLat: Double? = null,
        originLng: Double? = null,
        onFinished: (() -> Unit)? = null
    ) {
        val meta = loadDemoTrip() ?: return
        isPlaying = true
        var index = 0
        val baseNs = System.nanoTime()

        val dLat = if (originLat != null) originLat - meta.startLat else 0.0
        val dLng = if (originLng != null) originLng - meta.startLng else 0.0

        playbackRunnable = object : Runnable {
            override fun run() {
                if (!isPlaying || index >= meta.samples.size) {
                    isPlaying = false
                    onFinished?.invoke()
                    return
                }

                val pt = meta.samples[index]
                val currentNs = baseNs + (pt.sec * 1_000_000_000L)
                val effectiveLat = pt.lat + dLat
                val effectiveLng = pt.lng + dLng

                // 1. Deliver GNSS fix if GNSS is active
                if (pt.gnss && !engine.isGnssOutageSimulated) {
                    val gnss = GnssSample(
                        timestampNs = currentNs,
                        latitude = effectiveLat,
                        longitude = effectiveLng,
                        speedMps = pt.spd,
                        accuracyM = 4.0,
                        valid = true
                    )
                    engine.processGnss(gnss)
                }

                // 2. Deliver IMU sample to pipeline
                val imu = ImuSample(
                    timestampNs = currentNs,
                    accelX = pt.ax,
                    accelY = pt.ay,
                    accelZ = pt.az,
                    gyroX = pt.gx,
                    gyroY = pt.gy,
                    gyroZ = pt.gz
                )
                engine.processImu(imu)

                index++
                handler.postDelayed(this, 1000L / meta.sampleRateHz)
            }
        }

        handler.post(playbackRunnable!!)
    }

    fun stopPlayback() {
        isPlaying = false
        playbackRunnable?.let { handler.removeCallbacks(it) }
    }
}
