package com.rohit.georeckon.auth

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import com.google.firebase.auth.*
import com.rohit.georeckon.R
import com.rohit.georeckon.dashboard.DashboardActivity

/**
 * EmailAuthActivity — Firebase Email/Password Authentication.
 *
 * Supports:
 *  - Sign In  (default mode)
 *  - Sign Up  (toggled via tv_toggle link; shows Name field)
 *
 * On success → navigates to DashboardActivity with cleared back-stack.
 */
class EmailAuthActivity : ComponentActivity() {

    private lateinit var auth: FirebaseAuth
    private var isSignUp = false

    private lateinit var tvTitle: TextView
    private lateinit var tvSubtitle: TextView
    private lateinit var nameContainer: LinearLayout
    private lateinit var etName: EditText
    private lateinit var etEmail: EditText
    private lateinit var etPassword: EditText
    private lateinit var btnAuth: LinearLayout
    private lateinit var tvBtnText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var tvToggle: TextView

    // ── Lifecycle ──────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_email_auth)

        auth = FirebaseAuth.getInstance()

        bindViews()
        setupClickListeners()
    }

    // ── View binding ───────────────────────────────────────────────────────────
    private fun bindViews() {
        tvTitle       = findViewById(R.id.tv_title)
        tvSubtitle    = findViewById(R.id.tv_subtitle)
        nameContainer = findViewById(R.id.name_container)
        etName        = findViewById(R.id.et_name)
        etEmail       = findViewById(R.id.et_email)
        etPassword    = findViewById(R.id.et_password)
        btnAuth       = findViewById(R.id.btn_auth)
        tvBtnText     = findViewById(R.id.tv_btn_text)
        progressBar   = findViewById(R.id.progress_bar)
        tvToggle      = findViewById(R.id.tv_toggle)
    }

    private fun setupClickListeners() {
        findViewById<ImageButton>(R.id.btn_back).setOnClickListener { finish() }
        btnAuth.setOnClickListener  { handleAuth() }
        tvToggle.setOnClickListener { toggleMode() }
    }

    // ── Auth logic ─────────────────────────────────────────────────────────────
    private fun handleAuth() {
        val email    = etEmail.text.toString().trim()
        val password = etPassword.text.toString()

        if (!validateInputs(email, password)) return
        setLoading(true)

        if (isSignUp) {
            auth.createUserWithEmailAndPassword(email, password)
                .addOnCompleteListener(this) { task ->
                    setLoading(false)
                    if (task.isSuccessful) {
                        navigateToDashboard()
                    } else {
                        showError(task.exception)
                    }
                }
        } else {
            auth.signInWithEmailAndPassword(email, password)
                .addOnCompleteListener(this) { task ->
                    setLoading(false)
                    if (task.isSuccessful) {
                        navigateToDashboard()
                    } else {
                        showError(task.exception)
                    }
                }
        }
    }

    // ── Validation ─────────────────────────────────────────────────────────────
    private fun validateInputs(email: String, password: String): Boolean {
        if (email.isEmpty() ||
            !android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()
        ) {
            etEmail.error = "Enter a valid email address"
            etEmail.requestFocus()
            return false
        }
        if (password.length < 6) {
            etPassword.error = "Password must be at least 6 characters"
            etPassword.requestFocus()
            return false
        }
        return true
    }

    // ── Error handling ─────────────────────────────────────────────────────────
    private fun showError(exception: Exception?) {
        val msg = when (exception) {
            is FirebaseAuthWeakPasswordException     -> "Password too weak. Use at least 6 characters."
            is FirebaseAuthInvalidCredentialsException -> "Incorrect email or password."
            is FirebaseAuthUserCollisionException    -> "Account already exists. Try signing in."
            is FirebaseAuthInvalidUserException      -> "No account found with this email."
            else -> exception?.message ?: "Authentication failed. Please try again."
        }
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    // ── Toggle Sign In ↔ Sign Up ───────────────────────────────────────────────
    private fun toggleMode() {
        isSignUp = !isSignUp
        if (isSignUp) {
            tvTitle.text    = "Create Account"
            tvSubtitle.text = "Sign up to get started with GEO Reckon"
            tvBtnText.text  = "Create Account"
            tvToggle.text   = "Already have an account? Sign In"
            nameContainer.visibility = View.VISIBLE
        } else {
            tvTitle.text    = "Sign In"
            tvSubtitle.text = "Welcome back! Enter your credentials."
            tvBtnText.text  = "Sign In"
            tvToggle.text   = "Don't have an account? Sign Up"
            nameContainer.visibility = View.GONE
        }
    }

    // ── UI helpers ─────────────────────────────────────────────────────────────
    private fun setLoading(loading: Boolean) {
        btnAuth.isEnabled = !loading
        progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        tvBtnText.visibility   = if (loading) View.GONE   else View.VISIBLE
    }

    private fun navigateToDashboard() {
        startActivity(Intent(this, com.rohit.georeckon.dashboard.DashboardActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
        finish()
    }
}
