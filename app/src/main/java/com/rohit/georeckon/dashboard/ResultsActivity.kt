package com.rohit.georeckon.dashboard

import android.content.Intent
import android.os.Bundle
import android.widget.FrameLayout
import android.widget.LinearLayout
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import com.rohit.georeckon.R

class ResultsActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_results)

        findViewById<FrameLayout>(R.id.btn_results_back)?.setOnClickListener {
            finish()
        }

        findViewById<LinearLayout>(R.id.btn_results_track_ride)?.setOnClickListener {
            val intent = Intent(this, TrackRideActivity::class.java)
            startActivity(intent)
            finish()
        }
    }
}
