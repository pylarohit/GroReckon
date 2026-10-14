package com.rohit.georeckon.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.rohit.georeckon.data.model.ImuSample

/**
 * Manages live Android 6-axis IMU (Accelerometer + Gyroscope) acquisition.
 */
class ImuSensorManager(context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private var lastAccX = 0.0
    private var lastAccY = 0.0
    private var lastAccZ = 9.81

    private var lastGyroX = 0.0
    private var lastGyroY = 0.0
    private var lastGyroZ = 0.0

    private var listener: ((ImuSample) -> Unit)? = null
    var isListening = false
        private set

    fun startListening(onImuSample: (ImuSample) -> Unit) {
        listener = onImuSample
        accelerometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyroscope?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        isListening = true
    }

    fun stopListening() {
        sensorManager.unregisterListener(this)
        isListening = false
    }

    override fun onSensorChanged(event: SensorEvent) {
        val timestampNs = event.timestamp

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            lastAccX = event.values[0].toDouble()
            lastAccY = event.values[1].toDouble()
            lastAccZ = event.values[2].toDouble()
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            lastGyroX = event.values[0].toDouble()
            lastGyroY = event.values[1].toDouble()
            lastGyroZ = event.values[2].toDouble()
        }

        val sample = ImuSample(
            timestampNs = timestampNs,
            accelX = lastAccX,
            accelY = lastAccY,
            accelZ = lastAccZ,
            gyroX = lastGyroX,
            gyroY = lastGyroY,
            gyroZ = lastGyroZ
        )
        listener?.invoke(sample)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
