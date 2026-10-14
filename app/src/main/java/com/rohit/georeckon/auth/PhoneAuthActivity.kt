package com.rohit.georeckon.auth

import android.content.Intent
import android.os.Bundle
import android.os.CountDownTimer
import android.view.View
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.core.view.WindowCompat
import com.google.firebase.FirebaseException
import com.google.firebase.auth.*
import com.rohit.georeckon.R
import com.rohit.georeckon.dashboard.DashboardActivity
import java.util.concurrent.TimeUnit

/**
 * PhoneAuthActivity — Firebase Phone Authentication with OTP.
 *
 * Phase 1: User enters phone number → tap "Send OTP"
 * Phase 2: Firebase sends SMS → User enters 6-digit OTP → tap "Verify & Sign In"
 */
class PhoneAuthActivity : ComponentActivity() {

    private lateinit var auth: FirebaseAuth
    private var storedVerificationId: String? = null
    private lateinit var resendToken: PhoneAuthProvider.ForceResendingToken

    // Views
    private lateinit var tvTitle: TextView
    private lateinit var tvSubtitle: TextView
    private lateinit var sectionPhone: LinearLayout
    private lateinit var sectionOtp: LinearLayout
    private lateinit var etPhone: EditText
    private lateinit var etOtp: EditText
    private lateinit var btnSend: LinearLayout
    private lateinit var btnVerify: LinearLayout
    private lateinit var tvBtnSendText: TextView
    private lateinit var tvBtnVerifyText: TextView
    private lateinit var progressSend: ProgressBar
    private lateinit var progressVerify: ProgressBar
    private lateinit var tvResend: TextView
    private lateinit var tvPhoneDisplay: TextView

    // ── Lifecycle ──────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_phone_auth)

        auth = FirebaseAuth.getInstance()

        bindViews()
        setupClickListeners()
    }

    // ── View binding ───────────────────────────────────────────────────────────
    private fun bindViews() {
        tvTitle        = findViewById(R.id.tv_title)
        tvSubtitle     = findViewById(R.id.tv_subtitle)
        sectionPhone   = findViewById(R.id.section_phone)
        sectionOtp     = findViewById(R.id.section_otp)
        etPhone        = findViewById(R.id.et_phone)
        etOtp          = findViewById(R.id.et_otp)
        btnSend        = findViewById(R.id.btn_send_otp)
        btnVerify      = findViewById(R.id.btn_verify_otp)
        tvBtnSendText  = findViewById(R.id.tv_btn_send_text)
        tvBtnVerifyText= findViewById(R.id.tv_btn_verify_text)
        progressSend   = findViewById(R.id.progress_send)
        progressVerify = findViewById(R.id.progress_verify)
        tvResend       = findViewById(R.id.tv_resend)
        tvPhoneDisplay = findViewById(R.id.tv_phone_display)
    }

    private fun setupClickListeners() {
        findViewById<ImageButton>(R.id.btn_back).setOnClickListener { finish() }

        btnSend.setOnClickListener {
            val phone = etPhone.text.toString().trim()
            if (validatePhone(phone)) sendOtp(phone)
        }

        btnVerify.setOnClickListener {
            val otp = etOtp.text.toString().trim()
            if (otp.length == 6) verifyOtp(otp)
            else etOtp.error = "Enter the 6-digit code"
        }

        tvResend.setOnClickListener {
            resendOtp(etPhone.text.toString().trim())
        }
    }

    // ── Validation ─────────────────────────────────────────────────────────────
    private fun validatePhone(phone: String): Boolean {
        return when {
            phone.isEmpty() -> {
                etPhone.error = "Enter phone number"
                false
            }
            !phone.startsWith("+") -> {
                etPhone.error = "Include country code e.g. +91XXXXXXXXXX"
                false
            }
            phone.length < 10 -> {
                etPhone.error = "Number too short"
                false
            }
            else -> true
        }
    }

    // ── Firebase: Send OTP ─────────────────────────────────────────────────────
    private fun sendOtp(phoneNumber: String) {
        setLoading(btnSend, progressSend, tvBtnSendText, true)

        val options = PhoneAuthOptions.newBuilder(auth)
            .setPhoneNumber(phoneNumber)
            .setTimeout(60L, TimeUnit.SECONDS)
            .setActivity(this)
            .setCallbacks(verificationCallbacks)
            .build()

        PhoneAuthProvider.verifyPhoneNumber(options)
    }

    private fun resendOtp(phoneNumber: String) {
        val options = PhoneAuthOptions.newBuilder(auth)
            .setPhoneNumber(phoneNumber)
            .setTimeout(60L, TimeUnit.SECONDS)
            .setActivity(this)
            .setCallbacks(verificationCallbacks)
            .setForceResendingToken(resendToken)
            .build()

        PhoneAuthProvider.verifyPhoneNumber(options)
        tvResend.isEnabled = false
        tvResend.text = "Resending…"
    }

    // ── Firebase callbacks ─────────────────────────────────────────────────────
    private val verificationCallbacks =
        object : PhoneAuthProvider.OnVerificationStateChangedCallbacks() {

            /** Auto-retrieval succeeded (e.g. on Google-signed device) */
            override fun onVerificationCompleted(credential: PhoneAuthCredential) {
                signInWithCredential(credential)
            }

            /** Verification request failed */
            override fun onVerificationFailed(e: FirebaseException) {
                setLoading(btnSend, progressSend, tvBtnSendText, false)
                Toast.makeText(
                    this@PhoneAuthActivity,
                    "Failed to send OTP: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }

            /** SMS sent — show OTP input */
            override fun onCodeSent(
                verificationId: String,
                token: PhoneAuthProvider.ForceResendingToken
            ) {
                storedVerificationId = verificationId
                resendToken = token
                setLoading(btnSend, progressSend, tvBtnSendText, false)
                showOtpSection()
            }
        }

    // ── Firebase: Verify OTP ───────────────────────────────────────────────────
    private fun verifyOtp(otp: String) {
        val vId = storedVerificationId ?: run {
            Toast.makeText(this, "Request OTP first", Toast.LENGTH_SHORT).show()
            return
        }
        setLoading(btnVerify, progressVerify, tvBtnVerifyText, true)
        val credential = PhoneAuthProvider.getCredential(vId, otp)
        signInWithCredential(credential)
    }

    private fun signInWithCredential(credential: PhoneAuthCredential) {
        auth.signInWithCredential(credential)
            .addOnCompleteListener(this) { task ->
                setLoading(btnVerify, progressVerify, tvBtnVerifyText, false)
                if (task.isSuccessful) {
                    navigateToDashboard()
                } else {
                    val msg = when (task.exception) {
                        is FirebaseAuthInvalidCredentialsException ->
                            "Wrong OTP. Please try again."
                        else -> task.exception?.message ?: "Verification failed"
                    }
                    Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                }
            }
    }

    // ── UI helpers ─────────────────────────────────────────────────────────────
    private fun showOtpSection() {
        sectionPhone.visibility = View.GONE
        sectionOtp.visibility   = View.VISIBLE
        tvTitle.text     = "Enter OTP"
        tvSubtitle.text  = "Check your SMS messages"
        tvPhoneDisplay.text = "Code sent to ${etPhone.text}"
        etOtp.requestFocus()
        startResendTimer()
    }

    private fun startResendTimer() {
        tvResend.isEnabled = false
        object : CountDownTimer(60_000L, 1_000L) {
            override fun onTick(ms: Long)  { tvResend.text = "Resend OTP in ${ms / 1000}s" }
            override fun onFinish()        { tvResend.text = "Resend OTP"; tvResend.isEnabled = true }
        }.start()
    }

    private fun setLoading(
        btn: LinearLayout, progress: ProgressBar, label: TextView, loading: Boolean
    ) {
        btn.isEnabled = !loading
        progress.visibility = if (loading) View.VISIBLE else View.GONE
        label.visibility    = if (loading) View.GONE   else View.VISIBLE
    }

    private fun navigateToDashboard() {
        startActivity(Intent(this, com.rohit.georeckon.dashboard.DashboardActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
        finish()
    }
}
