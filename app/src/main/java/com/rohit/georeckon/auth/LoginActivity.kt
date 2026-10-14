package com.rohit.georeckon.auth

import android.content.Intent
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import coil3.ImageLoader
import coil3.gif.AnimatedImageDecoder
import coil3.gif.GifDecoder
import coil3.load
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInClient
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.common.api.ApiException
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.GoogleAuthProvider
import com.rohit.georeckon.R
import com.rohit.georeckon.dashboard.DashboardActivity

/**
 * LoginActivity — Entry point with three sign-in methods:
 *   1. Phone Number (OTP)  → PhoneAuthActivity
 *   2. Google              → Firebase GoogleAuthProvider
 *   3. Email / Password    → EmailAuthActivity
 *   4. Guest               → DashboardActivity (no account)
 */
class LoginActivity : ComponentActivity() {

    private lateinit var auth: FirebaseAuth
    private lateinit var googleSignInClient: GoogleSignInClient

    // ActivityResultLauncher for Google Sign-In intent
    private val googleSignInLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val task = GoogleSignIn.getSignedInAccountFromIntent(result.data)
        try {
            val account = task.getResult(ApiException::class.java)!!
            firebaseAuthWithGoogle(account.idToken!!)
        } catch (e: ApiException) {
            Toast.makeText(this, "Google sign-in failed: ${e.statusCode}", Toast.LENGTH_SHORT).show()
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_login)

        auth = FirebaseAuth.getInstance()

        // Skip login if already authenticated
        if (auth.currentUser != null) {
            navigateToDashboard()
            return
        }

        // ── Insets ──────────────────────────────────────────────────────────
        val bottomCard = findViewById<LinearLayout>(R.id.ll_bottom_card)
        ViewCompat.setOnApplyWindowInsetsListener(bottomCard) { view, insets ->
            val navBar = insets.getInsets(WindowInsetsCompat.Type.navigationBars())
            view.setPadding(
                view.paddingLeft,
                view.paddingTop,
                view.paddingRight,
                resources.getDimensionPixelSize(R.dimen.card_bottom_padding) + navBar.bottom
            )
            insets
        }

        // ── Coil image loader (GIF + PNG) ────────────────────────────────────
        val gifLoader = ImageLoader.Builder(this)
            .components {
                if (Build.VERSION.SDK_INT >= 28) {
                    add(AnimatedImageDecoder.Factory())
                } else {
                    add(GifDecoder.Factory())
                }
            }
            .build()

        findViewById<ImageView>(R.id.iv_hero_gif)
            .load("file:///android_asset/mapp.gif", gifLoader)
        findViewById<ImageView>(R.id.iv_google_icon)
            .load("file:///android_asset/google.png", gifLoader)
        findViewById<ImageView>(R.id.iv_mail_icon)
            .load("file:///android_asset/mail.png", gifLoader)

        // ── Google Sign-In client setup ──────────────────────────────────────
        val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestIdToken(getString(R.string.default_web_client_id)) // from google-services.json
            .requestEmail()
            .build()
        googleSignInClient = GoogleSignIn.getClient(this, gso)

        // ── Terms span ───────────────────────────────────────────────────────
        val grey  = Color.parseColor("#FF6C757D")
        val green = Color.parseColor("#FF4CAF50")
        val terms = SpannableStringBuilder().apply {
            append("By signing in, you agree to our ")
            setSpan(ForegroundColorSpan(grey), 0, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            val tStart = length; append("Terms")
            setSpan(ForegroundColorSpan(green), tStart, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            val aStart = length; append(" and ")
            setSpan(ForegroundColorSpan(grey), aStart, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            val pStart = length; append("Privacy Policy")
            setSpan(ForegroundColorSpan(green), pStart, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        findViewById<TextView>(R.id.tv_terms).text = terms

        // ── Button click listeners ───────────────────────────────────────────
        // Phone Number → PhoneAuthActivity
        findViewById<LinearLayout>(R.id.btn_phone).setOnClickListener {
            startActivity(Intent(this, PhoneAuthActivity::class.java))
        }

        // Google → Firebase Google Sign-In
        findViewById<LinearLayout>(R.id.btn_google).setOnClickListener {
            googleSignInLauncher.launch(googleSignInClient.signInIntent)
        }

        // Email → EmailAuthActivity
        findViewById<LinearLayout>(R.id.btn_email).setOnClickListener {
            startActivity(Intent(this, EmailAuthActivity::class.java))
        }

        // Guest → skip to dashboard
        findViewById<LinearLayout>(R.id.btn_guest).setOnClickListener {
            navigateToDashboard()
        }
    }

    // ── Firebase Google Auth ───────────────────────────────────────────────────
    private fun firebaseAuthWithGoogle(idToken: String) {
        val credential = GoogleAuthProvider.getCredential(idToken, null)
        auth.signInWithCredential(credential)
            .addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    navigateToDashboard()
                } else {
                    Toast.makeText(
                        this,
                        "Firebase auth failed: ${task.exception?.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
    }

    // ── Navigation ─────────────────────────────────────────────────────────────
    private fun navigateToDashboard() {
        startActivity(Intent(this, DashboardActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
        finish()
    }
}
