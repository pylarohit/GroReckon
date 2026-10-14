package com.rohit.georeckon.dashboard

import android.os.Bundle
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import com.rohit.georeckon.R

class DocumentationActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_documentation)

        findViewById<FrameLayout>(R.id.btn_doc_back)?.setOnClickListener {
            finish()
        }
    }
}
