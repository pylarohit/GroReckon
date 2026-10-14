package com.rohit.georeckon.dashboard

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import com.google.firebase.auth.FirebaseAuth
import com.rohit.georeckon.R
import com.rohit.georeckon.auth.LoginActivity

class DashboardActivity : ComponentActivity() {

    private lateinit var auth: FirebaseAuth
    private lateinit var tvHeaderAvatarInitials: TextView
    private lateinit var tvWelcomeTitle: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_dashboard)

        auth = FirebaseAuth.getInstance()

        initViews()
        setupUserAvatar()
        setupActions()
    }

    private fun initViews() {
        tvHeaderAvatarInitials = findViewById(R.id.tv_header_avatar_initials)
        tvWelcomeTitle = findViewById(R.id.tv_welcome_title)
    }

    private fun setupUserAvatar() {
        val user = auth.currentUser
        val displayName = user?.displayName ?: user?.email ?: "Commander"
        val firstName = displayName.split(" ", "@").firstOrNull()?.replaceFirstChar { it.uppercase() } ?: "Commander"
        tvWelcomeTitle.text = "Welcome, $firstName 👋"

        val initials = displayName.split(" ", "@", ".").filter { it.isNotBlank() }
            .take(2).map { it.first().uppercase() }.joinToString("")
        tvHeaderAvatarInitials.text = if (initials.isNotEmpty()) initials else "GR"

        findViewById<FrameLayout>(R.id.btn_header_profile)?.setOnClickListener {
            showAccountDialog()
        }
    }

    private fun setupActions() {
        // ── 1. Track Ride Button ───────────────────────────────────────────
        findViewById<LinearLayout>(R.id.btn_track_ride)?.setOnClickListener {
            val intent = Intent(this, TrackRideActivity::class.java)
            startActivity(intent)
        }

        // ── 2. Results Button ──────────────────────────────────────────────
        findViewById<LinearLayout>(R.id.btn_results)?.setOnClickListener {
            val intent = Intent(this, ResultsActivity::class.java)
            startActivity(intent)
        }

        // ── 3. Documentation Button ────────────────────────────────────────
        findViewById<LinearLayout>(R.id.btn_documentation)?.setOnClickListener {
            val intent = Intent(this, DocumentationActivity::class.java)
            startActivity(intent)
        }
    }

    private fun showAccountDialog() {
        val userEmail = auth.currentUser?.email ?: auth.currentUser?.phoneNumber ?: "Guest Mode"
        AlertDialog.Builder(this)
            .setTitle("Account & Fleet")
            .setMessage("Signed in as:\n$userEmail\n\nActive Fleet: 12 Vehicles\nKinoNet-R2 AI Engine: Ready\n\nSign out?")
            .setPositiveButton("Sign Out") { _, _ ->
                auth.signOut()
                startActivity(Intent(this, LoginActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                })
                finish()
            }
            .setNegativeButton("Close", null)
            .show()
    }
}
