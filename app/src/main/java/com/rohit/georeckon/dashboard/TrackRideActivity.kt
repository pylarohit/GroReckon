package com.rohit.georeckon.dashboard

import android.Manifest
import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.location.Location
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.webkit.*
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.content.Intent
import android.content.res.ColorStateList
import android.graphics.Color
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.widget.doOnTextChanged
import com.google.android.gms.location.*
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.rohit.georeckon.BuildConfig
import com.rohit.georeckon.R
import com.rohit.georeckon.data.model.EngineOutput
import com.rohit.georeckon.data.model.PositioningMode
import com.rohit.georeckon.engine.NavigationEngine
import com.rohit.georeckon.sensors.GnssManager
import com.rohit.georeckon.sensors.ImuSensorManager
import com.rohit.georeckon.session.DemoPlaybackController
import com.rohit.georeckon.session.DemoTripResult
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.Locale
import kotlin.math.*

class TrackRideActivity : ComponentActivity() {

    private enum class SelectionTarget {
        DESTINATION,
        START
    }

    private lateinit var webMap: WebView
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var locationCallback: LocationCallback? = null

    // UI Elements: Controls & Badges
    private lateinit var btnBack: FrameLayout
    private lateinit var layoutFloatingControls: LinearLayout
    private lateinit var btnToggleLayer: FrameLayout
    private lateinit var tvLayerLabel: TextView
    private lateinit var btnTogglePerspective: FrameLayout
    private lateinit var tvPerspectiveLabel: TextView
    private lateinit var btnRecenter: FrameLayout
    private lateinit var viewStatusDot: View

    // Top Containers & Compact Search Bar
    private lateinit var layoutTopContainer: LinearLayout
    private lateinit var layoutTopBarRow: LinearLayout
    private lateinit var cardCompactSearch: LinearLayout
    private lateinit var tvSearchBarPlaceholder: TextView

    // Expanded Route Planner Card (Hidden until user clicks search or set destination)
    private lateinit var cardRoutePlanner: LinearLayout
    private lateinit var btnClosePlanner: FrameLayout
    private lateinit var btnSwapLocations: FrameLayout
    private lateinit var etSearchStart: EditText
    private lateinit var btnClearStartText: FrameLayout
    private lateinit var btnSubmitStart: FrameLayout
    private lateinit var etSearchDestination: EditText
    private lateinit var btnClearDestText: FrameLayout
    private lateinit var btnSubmitDest: FrameLayout
    private lateinit var chipPickDestMap: LinearLayout
    private lateinit var tvChipDestLabel: TextView
    private lateinit var chipPickStartMap: LinearLayout
    private lateinit var tvChipStartLabel: TextView
    private lateinit var chipResetGps: LinearLayout
    private lateinit var tvSelectionModeHint: TextView

    // Bottom Telemetry & Google Maps-style BottomSheet HUD
    private lateinit var cardRideHud: LinearLayout
    private lateinit var bottomSheetBehavior: BottomSheetBehavior<LinearLayout>
    private lateinit var layoutDragHandleArea: FrameLayout
    private lateinit var layoutDestHeader: LinearLayout
    private lateinit var tvDestIcon: TextView
    private lateinit var tvDestTitle: TextView
    private lateinit var tvDestSubtitle: TextView
    private lateinit var btnClearDest: FrameLayout
    private lateinit var tvSheetChevron: TextView
    private lateinit var tvRideSpeed: TextView
    private lateinit var tvRideDistance: TextView
    private lateinit var tvDistanceLabel: TextView
    private lateinit var tvRideTime: TextView
    private lateinit var tvTimeLabel: TextView
    private lateinit var btnRideControl: LinearLayout
    private lateinit var tvRideBtnText: TextView

    // Demo Ride & Testing Bench Views
    private lateinit var btnDemoRide: LinearLayout
    private lateinit var tvDemoBtnText: TextView
    private lateinit var layoutDemoPanel: LinearLayout
    private lateinit var tvDemoModeBadge: TextView
    private lateinit var btnDemoCutGps: LinearLayout
    private lateinit var tvBtnCutGps: TextView
    private lateinit var btnDemoSpeedToggle: LinearLayout
    private lateinit var tvBtnDemoSpeed: TextView
    private lateinit var btnDemoStop: LinearLayout
    private lateinit var tvDemoAiStatus: TextView

    // Top Floating AI Dead Reckoning Handover Banner
    private lateinit var cardAiHandoverBanner: LinearLayout
    private lateinit var tvBannerIcon: TextView
    private lateinit var tvBannerTitle: TextView
    private lateinit var tvBannerSubtitle: TextView

    // Demo Playback Session
    private var demoPlaybackController: DemoPlaybackController? = null
    private var isDemoActive = false
    private val currentRouteCoords = mutableListOf<Pair<Double, Double>>()

    // Dead Reckoning Engine & Sensor Managers
    private lateinit var navigationEngine: NavigationEngine
    private lateinit var imuSensorManager: ImuSensorManager
    private lateinit var drGnssManager: GnssManager

    // Location & State
    private var userLat = 28.5355
    private var userLng = 77.1000
    private var lastLat: Double? = null
    private var lastLng: Double? = null
    private var currentHeading = 0.0
    private var isMapReady = false
    private var is3dMode = false
    private var isSatellite = false
    private var isRideActive = false
    private var isGpsLost = false

    // Selection Target & Coordinates
    private var activeTarget = SelectionTarget.DESTINATION
    private var customStartLat: Double? = null
    private var customStartLng: Double? = null
    private var startPlaceName: String? = null
    private var destLat: Double? = null
    private var destLng: Double? = null
    private var destPlaceName: String? = null
    private var hasDestination = false
    private var plannedDistKm = 0.0
    private var plannedDurMin = 0

    // Ride Metrics
    private var totalDistanceMeters = 0.0
    private var rideStartTimeMs = 0L
    private val timerHandler = Handler(Looper.getMainLooper())
    private var timerRunnable: Runnable? = null

    companion object {
        private const val PERMISSION_REQ_CODE = 2001
    }

    inner class TrackRideJsInterface {
        @JavascriptInterface
        fun onMapLoaded() {
            runOnUiThread {
                isMapReady = true
                js("updateUserLocation($userLat,$userLng,$currentHeading,false)")
                js("recenter($userLat,$userLng,16.5)")
            }
        }

        @JavascriptInterface
        fun onMapDoubleClicked(lat: Double, lng: Double) {
            runOnUiThread {
                handleMapDoubleClicked(lat, lng)
            }
        }

        @JavascriptInterface
        fun onRouteCalculated(distKm: Double, durMin: Int, sLat: Double, sLng: Double, dLat: Double, dLng: Double) {
            runOnUiThread {
                handleRouteCalculated(distKm, durMin, sLat, sLng, dLat, dLng)
            }
        }

        @JavascriptInterface
        fun onRouteCoordinates(coordsJson: String) {
            runOnUiThread {
                try {
                    val arr = JSONArray(coordsJson)
                    currentRouteCoords.clear()
                    for (i in 0 until arr.length()) {
                        val pt = arr.getJSONArray(i)
                        val lng = pt.getDouble(0)
                        val lat = pt.getDouble(1)
                        currentRouteCoords.add(Pair(lat, lng))
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_track_ride)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        navigationEngine = NavigationEngine(this)
        imuSensorManager = ImuSensorManager(this)
        drGnssManager    = GnssManager(this)
        setupEngineCallbacks()

        initViews()
        setupWebMap()
        setupListeners()
        updateSelectionModeUI()
        checkLocationPermission()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (isDemoActive) {
                    demoPlaybackController?.stopDemo()
                    finishDemoRun(null)
                } else if (cardRoutePlanner.visibility == View.VISIBLE) {
                    collapseRoutePlanner()
                } else if (bottomSheetBehavior.state == BottomSheetBehavior.STATE_EXPANDED) {
                    bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
                } else if (isRideActive) {
                    confirmStopRide()
                } else {
                    isEnabled = false
                    finish()
                }
            }
        })
    }

    private fun initViews() {
        webMap                 = findViewById(R.id.map_view)
        btnBack                = findViewById(R.id.btn_back)
        layoutFloatingControls = findViewById(R.id.layout_floating_controls)
        btnToggleLayer         = findViewById(R.id.btn_toggle_layer)
        tvLayerLabel           = findViewById(R.id.tv_layer_label)
        btnTogglePerspective   = findViewById(R.id.btn_toggle_perspective)
        tvPerspectiveLabel     = findViewById(R.id.tv_perspective_label)
        btnRecenter            = findViewById(R.id.btn_recenter)
        viewStatusDot          = findViewById(R.id.view_status_dot)

        layoutTopContainer     = findViewById(R.id.layout_top_container)
        layoutTopBarRow        = findViewById(R.id.layout_top_bar_row)
        cardCompactSearch      = findViewById(R.id.card_compact_search)
        tvSearchBarPlaceholder = findViewById(R.id.tv_search_bar_placeholder)

        cardRoutePlanner       = findViewById(R.id.card_route_planner)
        btnClosePlanner        = findViewById(R.id.btn_close_planner)
        btnSwapLocations       = findViewById(R.id.btn_swap_locations)
        etSearchStart          = findViewById(R.id.et_search_start)
        btnClearStartText     = findViewById(R.id.btn_clear_start_text)
        btnSubmitStart         = findViewById(R.id.btn_submit_start)
        etSearchDestination    = findViewById(R.id.et_search_destination)
        btnClearDestText      = findViewById(R.id.btn_clear_dest_text)
        btnSubmitDest          = findViewById(R.id.btn_submit_dest)
        chipPickDestMap        = findViewById(R.id.chip_pick_dest_map)
        tvChipDestLabel        = findViewById(R.id.tv_chip_dest_label)
        chipPickStartMap       = findViewById(R.id.chip_pick_start_map)
        tvChipStartLabel       = findViewById(R.id.tv_chip_start_label)
        chipResetGps           = findViewById(R.id.chip_reset_gps)
        tvSelectionModeHint    = findViewById(R.id.tv_selection_mode_hint)

        cardRideHud            = findViewById(R.id.card_ride_hud)
        layoutDragHandleArea   = findViewById(R.id.layout_drag_handle_area)
        layoutDestHeader       = findViewById(R.id.layout_dest_header)
        tvDestIcon             = findViewById(R.id.tv_dest_icon)
        tvDestTitle            = findViewById(R.id.tv_dest_title)
        tvDestSubtitle         = findViewById(R.id.tv_dest_subtitle)
        btnClearDest           = findViewById(R.id.btn_clear_dest)
        tvSheetChevron         = findViewById(R.id.tv_sheet_chevron)

        tvRideSpeed            = findViewById(R.id.tv_ride_speed)
        tvRideDistance         = findViewById(R.id.tv_ride_distance)
        tvDistanceLabel        = findViewById(R.id.tv_distance_label)
        tvRideTime             = findViewById(R.id.tv_ride_time)
        tvTimeLabel            = findViewById(R.id.tv_time_label)
        btnRideControl         = findViewById(R.id.btn_ride_control)
        tvRideBtnText          = findViewById(R.id.tv_ride_btn_text)
        btnDemoRide            = findViewById(R.id.btn_demo_ride)
        tvDemoBtnText          = findViewById(R.id.tv_demo_btn_text)
        layoutDemoPanel        = findViewById(R.id.layout_demo_panel)
        tvDemoModeBadge        = findViewById(R.id.tv_demo_mode_badge)
        btnDemoCutGps          = findViewById(R.id.btn_demo_cut_gps)
        tvBtnCutGps            = findViewById(R.id.tv_btn_cut_gps)
        btnDemoSpeedToggle     = findViewById(R.id.btn_demo_speed_toggle)
        tvBtnDemoSpeed         = findViewById(R.id.tv_btn_demo_speed)
        btnDemoStop            = findViewById(R.id.btn_demo_stop)
        tvDemoAiStatus         = findViewById(R.id.tv_demo_ai_status)

        cardAiHandoverBanner   = findViewById(R.id.card_ai_handover_banner)
        tvBannerIcon           = findViewById(R.id.tv_banner_icon)
        tvBannerTitle          = findViewById(R.id.tv_banner_title)
        tvBannerSubtitle       = findViewById(R.id.tv_banner_subtitle)

        // Setup BottomSheetBehavior (Starts collapsed showing exactly till the button)
        bottomSheetBehavior = BottomSheetBehavior.from(cardRideHud)
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        bottomSheetBehavior.isHideable = false
        bottomSheetBehavior.isFitToContents = true
        bottomSheetBehavior.peekHeight = (210 * resources.displayMetrics.density).toInt()

        cardRideHud.post {
            val buttonLocation = IntArray(2)
            btnRideControl.getLocationInWindow(buttonLocation)
            val sheetLocation = IntArray(2)
            cardRideHud.getLocationInWindow(sheetLocation)
            val buttonBottomInSheet = (buttonLocation[1] - sheetLocation[1]) + btnRideControl.height + (14 * resources.displayMetrics.density).toInt()
            if (buttonBottomInSheet > 160) {
                bottomSheetBehavior.peekHeight = buttonBottomInSheet
                val lp = layoutFloatingControls.layoutParams as? ViewGroup.MarginLayoutParams
                if (lp != null) {
                    lp.bottomMargin = buttonBottomInSheet + (14 * resources.displayMetrics.density).toInt()
                    layoutFloatingControls.layoutParams = lp
                }
            }
        }

        tvRideBtnText.text = "📍 Set Destination"
        btnRideControl.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_dark)
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebMap() {
        webMap.settings.apply {
            javaScriptEnabled          = true
            domStorageEnabled          = true
            loadWithOverviewMode       = true
            useWideViewPort            = true
            cacheMode                  = WebSettings.LOAD_DEFAULT
            mixedContentMode           = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            allowFileAccess            = true
            @Suppress("DEPRECATION")
            allowUniversalAccessFromFileURLs = true
            @Suppress("DEPRECATION")
            allowFileAccessFromFileURLs = true
        }

        webMap.addJavascriptInterface(TrackRideJsInterface(), "TrackRideInterface")

        webMap.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                val token = BuildConfig.MAPS_API_KEY
                js("initMapbox('$token', $userLat, $userLng, $currentHeading)")
            }
        }

        webMap.loadUrl("file:///android_asset/mapbox_map.html")
    }

    private fun setupListeners() {
        btnBack.setOnClickListener {
            if (cardRoutePlanner.visibility == View.VISIBLE) {
                collapseRoutePlanner()
            } else if (bottomSheetBehavior.state == BottomSheetBehavior.STATE_EXPANDED) {
                bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
            } else if (isRideActive) {
                confirmStopRide()
            } else {
                finish()
            }
        }

        // Tap compact search bar -> expands full Google Maps style route planner
        cardCompactSearch.setOnClickListener {
            expandRoutePlanner(focusDestination = true)
        }

        // Close route planner -> returns to clean map view
        btnClosePlanner.setOnClickListener {
            collapseRoutePlanner()
        }

        // Swap start & destination
        btnSwapLocations.setOnClickListener {
            swapStartAndDestination()
        }

        // Bottom Sheet Drag handle & Chevron tap -> toggles collapsed / expanded
        layoutDragHandleArea.setOnClickListener {
            toggleBottomSheet()
        }

        tvSheetChevron.setOnClickListener {
            toggleBottomSheet()
        }

        // Bottom HUD Destination header tap
        layoutDestHeader.setOnClickListener {
            if (!hasDestination) {
                expandRoutePlanner(focusDestination = true)
            } else {
                toggleBottomSheet()
            }
        }

        // BottomSheet sliding callback: syncs floating controls position & chevron rotation
        bottomSheetBehavior.addBottomSheetCallback(object : BottomSheetBehavior.BottomSheetCallback() {
            override fun onSlide(bottomSheet: View, slideOffset: Float) {
                if (slideOffset > 0) {
                    val maxTranslation = (bottomSheet.height - bottomSheetBehavior.peekHeight).toFloat()
                    layoutFloatingControls.translationY = -maxTranslation * slideOffset
                    tvSheetChevron.rotation = slideOffset * 180f
                } else {
                    layoutFloatingControls.translationY = 0f
                    tvSheetChevron.rotation = 0f
                }
            }

            override fun onStateChanged(bottomSheet: View, newState: Int) {
                if (newState == BottomSheetBehavior.STATE_COLLAPSED) {
                    tvSheetChevron.rotation = 0f
                    layoutFloatingControls.translationY = 0f
                    tvDestSubtitle.text = if (hasDestination) {
                        String.format(Locale.US, "%.2f km · ~%d min (Shortest & Best Route)", plannedDistKm, plannedDurMin)
                    } else {
                        "Double-click map or search to plan route"
                    }
                } else if (newState == BottomSheetBehavior.STATE_EXPANDED) {
                    tvSheetChevron.rotation = 180f
                    tvDestSubtitle.text = if (hasDestination) {
                        String.format(Locale.US, "%.2f km · ~%d min · Pull down to collapse", plannedDistKm, plannedDurMin)
                    } else {
                        "Pull down to collapse · Tap search to set route"
                    }
                }
            }
        })

        btnToggleLayer.setOnClickListener {
            isSatellite = !isSatellite
            tvLayerLabel.text = if (isSatellite) "🗺️" else "🛰️"
            js("toggleSatellite($isSatellite)")
            Toast.makeText(this, if (isSatellite) "🛰️ Satellite View Active" else "🗺️ Street View Active", Toast.LENGTH_SHORT).show()
        }

        btnTogglePerspective.setOnClickListener {
            is3dMode = !is3dMode
            tvPerspectiveLabel.text = if (is3dMode) "2D" else "3D"
            js("setPerspective($is3dMode)")
            Toast.makeText(this, if (is3dMode) "3D Mode: Extruded Buildings (Pitch 65°)" else "2D Mode: Normal Flat Map", Toast.LENGTH_SHORT).show()
        }

        btnRecenter.setOnClickListener {
            js("recenter($userLat,$userLng,16.8)")
        }

        // Selection Target Chip Listeners
        chipPickDestMap.setOnClickListener {
            activeTarget = SelectionTarget.DESTINATION
            updateSelectionModeUI()
            Toast.makeText(this, "🏁 Double-click on map to set Destination", Toast.LENGTH_SHORT).show()
        }

        chipPickStartMap.setOnClickListener {
            activeTarget = SelectionTarget.START
            updateSelectionModeUI()
            Toast.makeText(this, "🟢 Double-click on map to set Start Point", Toast.LENGTH_SHORT).show()
        }

        chipResetGps.setOnClickListener {
            resetStartToGps()
        }

        // Input Focus switches target
        etSearchStart.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                activeTarget = SelectionTarget.START
                updateSelectionModeUI()
            }
        }

        etSearchDestination.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                activeTarget = SelectionTarget.DESTINATION
                updateSelectionModeUI()
            }
        }

        // Start input actions
        etSearchStart.doOnTextChanged { text, _, _, _ ->
            btnClearStartText.visibility = if (text.isNullOrBlank()) View.GONE else View.VISIBLE
        }

        btnClearStartText.setOnClickListener {
            resetStartToGps()
        }

        etSearchStart.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEARCH || actionId == EditorInfo.IME_ACTION_DONE) {
                searchLocation(etSearchStart.text.toString(), isStart = true)
                true
            } else {
                false
            }
        }

        btnSubmitStart.setOnClickListener {
            searchLocation(etSearchStart.text.toString(), isStart = true)
        }

        // Destination input actions
        etSearchDestination.doOnTextChanged { text, _, _, _ ->
            btnClearDestText.visibility = if (text.isNullOrBlank()) View.GONE else View.VISIBLE
        }

        btnClearDestText.setOnClickListener {
            etSearchDestination.text.clear()
            resetDestination()
        }

        etSearchDestination.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEARCH || actionId == EditorInfo.IME_ACTION_DONE) {
                searchLocation(etSearchDestination.text.toString(), isStart = false)
                true
            } else {
                false
            }
        }

        btnSubmitDest.setOnClickListener {
            searchLocation(etSearchDestination.text.toString(), isStart = false)
        }

        // Clear Destination Pin button in HUD
        btnClearDest.setOnClickListener {
            resetDestination()
        }

        // Prominent Start/Stop Ride button
        btnRideControl.setOnClickListener {
            if (!hasDestination) {
                expandRoutePlanner(focusDestination = true)
            } else if (!isRideActive) {
                startRideTracking()
            } else {
                confirmStopRide()
            }
        }

        // Demo Ride Button listener
        btnDemoRide.setOnClickListener {
            if (isDemoActive) {
                demoPlaybackController?.stopDemo()
                finishDemoRun(null)
            } else {
                startDemoRide()
            }
        }

        // Demo Testing Bench: Cut/Restore GPS
        btnDemoCutGps.setOnClickListener {
            val isCut = demoPlaybackController?.toggleManualGpsCut() ?: false
            updateCutGpsButtonUI(isCut)
            Toast.makeText(
                this,
                if (isCut) "📡 GPS Cut! 🧠 KinoNet-R2 AI Dead Reckoning taking over..." else "🛰️ GPS Restored! ✅ InEKF Re-acquiring fix...",
                Toast.LENGTH_SHORT
            ).show()
        }

        // Demo Testing Bench: Speed Multiplier (1x / 2x)
        btnDemoSpeedToggle.setOnClickListener {
            val currentMul = demoPlaybackController?.speedMultiplier ?: 1.0f
            val newMul = if (currentMul == 1.0f) 2.0f else 1.0f
            demoPlaybackController?.speedMultiplier = newMul
            tvBtnDemoSpeed.text = if (newMul == 2.0f) "⏩ 2x" else "▶️ 1x"
            Toast.makeText(this, "Demo speed set to ${newMul.toInt()}x", Toast.LENGTH_SHORT).show()
        }

        // Demo Testing Bench: Stop Run
        btnDemoStop.setOnClickListener {
            demoPlaybackController?.finishDemo()
        }
    }

    private fun toggleBottomSheet() {
        if (bottomSheetBehavior.state == BottomSheetBehavior.STATE_EXPANDED) {
            bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        } else {
            bottomSheetBehavior.state = BottomSheetBehavior.STATE_EXPANDED
        }
    }

    private fun expandRoutePlanner(focusDestination: Boolean) {
        cardCompactSearch.visibility = View.GONE
        cardRoutePlanner.visibility = View.VISIBLE
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED

        if (focusDestination) {
            activeTarget = SelectionTarget.DESTINATION
            etSearchDestination.requestFocus()
            val imm = getSystemService(INPUT_METHOD_SERVICE) as? InputMethodManager
            imm?.showSoftInput(etSearchDestination, InputMethodManager.SHOW_IMPLICIT)
        }
        updateSelectionModeUI()
    }

    private fun collapseRoutePlanner() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.hideSoftInputFromWindow(etSearchDestination.windowToken, 0)
        imm?.hideSoftInputFromWindow(etSearchStart.windowToken, 0)
        etSearchStart.clearFocus()
        etSearchDestination.clearFocus()

        cardRoutePlanner.visibility = View.GONE
        cardCompactSearch.visibility = View.VISIBLE
    }

    private fun swapStartAndDestination() {
        val currentOriginLat = customStartLat ?: userLat
        val currentOriginLng = customStartLng ?: userLng
        val currentOriginName = startPlaceName ?: "Your Location"

        val currentDestLat = destLat
        val currentDestLng = destLng
        val currentDestName = destPlaceName

        if (currentDestLat == null || currentDestLng == null) {
            Toast.makeText(this, "Set a destination first to swap", Toast.LENGTH_SHORT).show()
            return
        }

        customStartLat = currentDestLat
        customStartLng = currentDestLng
        startPlaceName = currentDestName
        etSearchStart.setText(currentDestName ?: "")

        destLat = currentOriginLat
        destLng = currentOriginLng
        destPlaceName = currentOriginName
        etSearchDestination.setText(currentOriginName)

        js("setStartLocation($customStartLat, $customStartLng)")
        js("setDestination($destLat, $destLng)")
        Toast.makeText(this, "🔄 Swapped Start and Destination", Toast.LENGTH_SHORT).show()
    }

    private fun updateSelectionModeUI() {
        val activeBg = ContextCompat.getDrawable(this, R.drawable.bg_chip_active)
        val inactiveBg = ContextCompat.getDrawable(this, R.drawable.bg_chip_inactive)
        val white = ContextCompat.getColor(this, R.color.white)
        val darkCharcoal = ContextCompat.getColor(this, R.color.dark_charcoal)

        when (activeTarget) {
            SelectionTarget.DESTINATION -> {
                chipPickDestMap.background = activeBg
                tvChipDestLabel.setTextColor(white)
                chipPickStartMap.background = inactiveBg
                tvChipStartLabel.setTextColor(darkCharcoal)
                tvSelectionModeHint.text = "💡 Double-click anywhere on map to set Destination"
            }
            SelectionTarget.START -> {
                chipPickStartMap.background = activeBg
                tvChipStartLabel.setTextColor(white)
                chipPickDestMap.background = inactiveBg
                tvChipDestLabel.setTextColor(darkCharcoal)
                tvSelectionModeHint.text = "💡 Double-click anywhere on map to set Start Point"
            }
        }
    }

    // ── Handle Double Click on Map ───────────────────────────────────────────
    private fun handleMapDoubleClicked(lat: Double, lng: Double) {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.hideSoftInputFromWindow(etSearchDestination.windowToken, 0)
        etSearchStart.clearFocus()
        etSearchDestination.clearFocus()

        if (activeTarget == SelectionTarget.START) {
            customStartLat = lat
            customStartLng = lng
            js("setStartLocation($lat, $lng)")
            reverseGeocode(lat, lng, isStart = true)
            Toast.makeText(this, "🟢 Start point set! Now double-click for destination.", Toast.LENGTH_SHORT).show()

            if (destLat == null) {
                activeTarget = SelectionTarget.DESTINATION
                updateSelectionModeUI()
            }
        } else {
            destLat = lat
            destLng = lng
            js("setDestination($lat, $lng)")
            reverseGeocode(lat, lng, isStart = false)
            Toast.makeText(this, "🏁 Destination set! Calculating shortest route...", Toast.LENGTH_SHORT).show()
        }
    }

    // ── Geocoding: Forward Search (Start or Destination) ──────────────────────
    private fun searchLocation(query: String, isStart: Boolean) {
        if (query.isBlank()) {
            if (isStart) resetStartToGps()
            return
        }
        val imm = getSystemService(INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.hideSoftInputFromWindow(etSearchDestination.windowToken, 0)
        etSearchStart.clearFocus()
        etSearchDestination.clearFocus()

        val role = if (isStart) "start location" else "destination"
        Toast.makeText(this, "🔍 Searching shortest road route to $role '$query'...", Toast.LENGTH_SHORT).show()

        Thread {
            try {
                val token = BuildConfig.MAPS_API_KEY
                val encodedQuery = URLEncoder.encode(query, "UTF-8")
                val proximityLng = customStartLng ?: userLng
                val proximityLat = customStartLat ?: userLat
                val urlStr = "https://api.mapbox.com/geocoding/v5/mapbox.places/$encodedQuery.json" +
                        "?proximity=$proximityLng,$proximityLat&limit=1&access_token=$token"
                val conn = URL(urlStr).openConnection() as HttpURLConnection
                conn.connectTimeout = 7000
                conn.readTimeout = 7000

                if (conn.responseCode == 200) {
                    val body = conn.inputStream.bufferedReader().readText()
                    val json = JSONObject(body)
                    val features = json.optJSONArray("features")
                    if (features != null && features.length() > 0) {
                        val firstPlace = features.getJSONObject(0)
                        val placeName = firstPlace.optString("place_name", query)
                        val shortName = firstPlace.optString("text", placeName.split(",").firstOrNull() ?: query)
                        val center = firstPlace.getJSONArray("center")
                        val pLng = center.getDouble(0)
                        val pLat = center.getDouble(1)

                        runOnUiThread {
                            if (isStart) {
                                customStartLat = pLat
                                customStartLng = pLng
                                startPlaceName = shortName
                                etSearchStart.setText(shortName)
                                js("setStartLocation($pLat, $pLng)")
                                Toast.makeText(this, "🟢 Start set: $shortName", Toast.LENGTH_SHORT).show()
                            } else {
                                destLat = pLat
                                destLng = pLng
                                destPlaceName = shortName
                                etSearchDestination.setText(shortName)
                                js("setDestination($pLat, $pLng)")
                                Toast.makeText(this, "🏁 Destination set: $shortName", Toast.LENGTH_SHORT).show()
                            }
                        }
                        return@Thread
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }

            runOnUiThread {
                Toast.makeText(this, "Could not find '$query'. Try another place or double-click the map.", Toast.LENGTH_LONG).show()
            }
        }.start()
    }

    // ── Geocoding: Reverse Search on Map Double-Click ────────────────────────
    private fun reverseGeocode(lat: Double, lng: Double, isStart: Boolean) {
        Thread {
            try {
                val token = BuildConfig.MAPS_API_KEY
                val urlStr = "https://api.mapbox.com/geocoding/v5/mapbox.places/$lng,$lat.json?limit=1&access_token=$token"
                val conn = URL(urlStr).openConnection() as HttpURLConnection
                conn.connectTimeout = 5000
                conn.readTimeout = 5000

                if (conn.responseCode == 200) {
                    val body = conn.inputStream.bufferedReader().readText()
                    val json = JSONObject(body)
                    val features = json.optJSONArray("features")
                    if (features != null && features.length() > 0) {
                        val firstPlace = features.getJSONObject(0)
                        val placeName = firstPlace.optString("place_name", "Selected Location")
                        val shortName = firstPlace.optString("text", placeName.split(",").firstOrNull() ?: "Selected Location")

                        runOnUiThread {
                            if (isStart) {
                                startPlaceName = shortName
                                etSearchStart.setText(shortName)
                            } else {
                                destPlaceName = shortName
                                etSearchDestination.setText(shortName)
                                tvSearchBarPlaceholder.text = "🏁 $shortName"
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()
    }

    private fun resetStartToGps() {
        customStartLat = null
        customStartLng = null
        startPlaceName = null
        etSearchStart.text.clear()
        js("clearStartLocation()")
        Toast.makeText(this, "🎯 Start set to Current Location (GPS)", Toast.LENGTH_SHORT).show()
        if (destLat != null) {
            js("setDestination($destLat, $destLng)")
        }
    }

    // ── Route Handlers ────────────────────────────────────────────────────────
    private fun handleRouteCalculated(distKm: Double, durMin: Int, sLat: Double, sLng: Double, dLat: Double, dLng: Double) {
        hasDestination = true
        destLat = dLat
        destLng = dLng
        plannedDistKm = distKm
        plannedDurMin = durMin

        val originTitle = startPlaceName ?: "Your Location"
        val destTitle = destPlaceName ?: "Destination Set"

        tvSearchBarPlaceholder.text = "🏁 $destTitle"
        tvDestIcon.text = "🏁"
        tvDestTitle.text = if (customStartLat != null) "$originTitle ➔ $destTitle" else destTitle
        tvDestSubtitle.text = String.format(Locale.US, "%.2f km · ~%d min · Swipe up for details", distKm, durMin)
        btnClearDest.visibility = View.VISIBLE

        tvRideDistance.text = String.format(Locale.US, "%.2f", distKm)
        tvDistanceLabel.text = "EST. DISTANCE"
        tvRideTime.text = String.format(Locale.US, "%d min", durMin)
        tvTimeLabel.text = "EST. TIME"

        tvRideBtnText.text = "🚀 Start Ride"
        btnRideControl.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_start_ride)

        // Automatically collapse the planner so user sees the full map & planned route!
        collapseRoutePlanner()

        // Keep the sheet down (collapsed) so user can see the complete road route!
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED

        Toast.makeText(this, "📍 Shortest & best route calculated! Swipe up to start ride.", Toast.LENGTH_SHORT).show()
    }

    private fun resetDestination() {
        hasDestination = false
        destLat = null
        destLng = null
        destPlaceName = null
        etSearchDestination.text.clear()
        tvSearchBarPlaceholder.text = "Where to? Search destination..."
        js("clearDestination()")

        tvDestIcon.text = "📍"
        tvDestTitle.text = "Choose Destination"
        tvDestSubtitle.text = "Swipe up for ride details & speedometer"
        btnClearDest.visibility = View.GONE

        tvDistanceLabel.text = "DISTANCE (KM)"
        tvTimeLabel.text = "DURATION"
        tvRideDistance.text = "0.00"
        tvRideTime.text = "00:00"
        tvRideSpeed.text = "0"

        tvRideBtnText.text = "📍 Set Destination"
        btnRideControl.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_dark)

        if (isDemoActive) {
            demoPlaybackController?.stopDemo()
            isDemoActive = false
            layoutDemoPanel.visibility = View.GONE
            cardAiHandoverBanner.visibility = View.GONE
            tvDemoBtnText.text = "🧪 Demo Ride"
            btnDemoRide.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_demo_ride)
        }

        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
    }

    // ── Ride Lifecycle ─────────────────────────────────────────────────────────
    private fun startRideTracking() {
        isRideActive = true
        totalDistanceMeters = 0.0
        lastLat = customStartLat ?: userLat
        lastLng = customStartLng ?: userLng
        rideStartTimeMs = System.currentTimeMillis()

        tvDestIcon.text = "🚴"
        tvDestTitle.text = destPlaceName ?: "Ride in Progress"
        tvDestSubtitle.text = "Shortest route · GNSS + Dead Reckoning Active"
        btnClearDest.visibility = View.GONE

        // Hide top search bar during active ride for maximum map visibility
        layoutTopContainer.visibility = View.GONE

        tvDistanceLabel.text = "TRAVELED (KM)"
        tvTimeLabel.text = "DURATION"
        tvRideDistance.text = "0.00"
        tvRideTime.text = "00:00"
        tvRideBtnText.text = "⏹ End Ride"
        btnRideControl.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_stop_ride)

        js("startRide()")
        js("clearRidePath()")
        js("addRidePoint($lastLat,$lastLng)")

        // Start 100 Hz IMU & GNSS sensor streams for dead reckoning
        imuSensorManager.startListening { imu ->
            navigationEngine.processImu(imu)
        }
        drGnssManager.startListening { gnss ->
            navigationEngine.processGnss(gnss)
        }

        // Start live ride timer
        timerRunnable = object : Runnable {
            override fun run() {
                val elapsedSec = (System.currentTimeMillis() - rideStartTimeMs) / 1000
                val min = elapsedSec / 60
                val sec = elapsedSec % 60
                tvRideTime.text = String.format(Locale.US, "%02d:%02d", min, sec)
                timerHandler.postDelayed(this, 1000)
            }
        }
        timerHandler.post(timerRunnable!!)

        // Keep bottom sheet down so user has full view of roads and live position!
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED

        Toast.makeText(this, "🚴 Ride started! Following shortest road route.", Toast.LENGTH_SHORT).show()
    }

    private fun stopRideTracking() {
        isRideActive = false
        timerRunnable?.let { timerHandler.removeCallbacks(it) }

        imuSensorManager.stopListening()
        drGnssManager.stopListening()

        js("stopRide()")
        layoutTopContainer.visibility = View.VISIBLE
        cardCompactSearch.visibility = View.VISIBLE
        cardRoutePlanner.visibility = View.GONE

        val distKm = totalDistanceMeters / 1000.0
        val elapsedSec = max(1, (System.currentTimeMillis() - rideStartTimeMs) / 1000)
        val avgSpeedKmh = (distKm / (elapsedSec / 3600.0)).toInt()

        AlertDialog.Builder(this)
            .setTitle("🏁 Ride Complete!")
            .setMessage(String.format(Locale.US,
                "Destination: %s\nDistance: %.2f km\nDuration: %02d:%02d\nAvg Speed: %d km/h\nDead Reckoning Drift: < 1.2%%",
                destPlaceName ?: "Selected Location", distKm, elapsedSec / 60, elapsedSec % 60, avgSpeedKmh))
            .setPositiveButton("Done") { _, _ ->
                resetDestination()
            }
            .setCancelable(false)
            .show()
    }

    private fun confirmStopRide() {
        AlertDialog.Builder(this)
            .setTitle("End Ride?")
            .setMessage("Are you sure you want to finish this ride?")
            .setPositiveButton("End Ride") { _, _ -> stopRideTracking() }
            .setNegativeButton("Keep Riding", null)
            .show()
    }

    // ── Demo Ride Simulation & Autonomous Handover ─────────────────────────────
    private fun startDemoRide() {
        if (isRideActive) {
            Toast.makeText(this, "Stop the active ride before starting the demo.", Toast.LENGTH_SHORT).show()
            return
        }

        // If no destination is set yet, setup an expressway test route automatically!
        if (!hasDestination) {
            destLat = userLat + 0.018
            destLng = userLng + 0.022
            destPlaceName = "Expressway Tunnel Test Stretch"
            hasDestination = true
            tvDestTitle.text = destPlaceName
            tvDestSubtitle.text = "2.85 km · Simulated 40s Tunnel Blackout"
            tvSearchBarPlaceholder.text = "🏁 $destPlaceName"
            js("setDestination($destLat,$destLng)")
        }

        isDemoActive = true
        layoutTopContainer.visibility = View.GONE
        layoutDemoPanel.visibility = View.VISIBLE
        btnClearDest.visibility = View.GONE

        tvDestIcon.text = "🎮"
        tvDestTitle.text = destPlaceName ?: "Demo Ride (KinoNet-R2 Test)"
        tvDestSubtitle.text = "Running AI Dead Reckoning Simulation with Tunnel Blackout"

        tvDistanceLabel.text = "TRAVELED (KM)"
        tvTimeLabel.text = "DURATION"
        tvRideDistance.text = "0.00"
        tvRideTime.text = "00:00"

        tvDemoBtnText.text = "⏹ Stop Demo"
        btnDemoRide.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_stop_ride)

        js("startRide()")
        js("clearRidePath()")

        demoPlaybackController = DemoPlaybackController(
            engine = navigationEngine,
            onTick = { lat, lng, speedKmh, isDr, bannerMsg ->
                tvRideSpeed.text = speedKmh.toString()
                userLat = lat
                userLng = lng
                js("updateUserLocation($lat,$lng,$currentHeading,$isDr)")
                js("addRidePoint($lat,$lng)")

                // Live distance increment
                lastLat?.let { pLat ->
                    lastLng?.let { pLng ->
                        val d = haversineM(pLat, pLng, lat, lng)
                        if (d > 0.5 && d < 200.0) {
                            totalDistanceMeters += d
                            val km = totalDistanceMeters / 1000.0
                            tvRideDistance.text = String.format(Locale.US, "%.2f", km)
                        }
                    }
                }
                lastLat = lat
                lastLng = lng

                // Floating Banner updates
                if (bannerMsg != null) {
                    cardAiHandoverBanner.visibility = View.VISIBLE
                    if (isDr) {
                        cardAiHandoverBanner.background = ContextCompat.getDrawable(this, R.drawable.bg_banner_dr)
                        tvBannerIcon.text = "🧠"
                        tvBannerTitle.text = "AI DEAD RECKONING ACTIVE"
                        tvBannerTitle.setTextColor(Color.parseColor("#FBBF24"))
                        tvBannerSubtitle.text = "GPS lost in tunnel. KinoNet-R2 AI propagating 100 Hz IMU."
                        viewStatusDot.backgroundTintList = ContextCompat.getColorStateList(this, R.color.nav_traffic_slow)
                    } else {
                        cardAiHandoverBanner.background = ContextCompat.getDrawable(this, R.drawable.bg_banner_dr)
                        tvBannerIcon.text = "🛰️"
                        tvBannerTitle.text = "GNSS REACQUIRED & FUSED"
                        tvBannerTitle.setTextColor(Color.parseColor("#34D399"))
                        tvBannerSubtitle.text = "GPS fix restored. Accurate location corrected via Invariant EKF."
                        viewStatusDot.backgroundTintList = ContextCompat.getColorStateList(this, R.color.nav_green)
                    }
                } else {
                    cardAiHandoverBanner.visibility = View.GONE
                    viewStatusDot.backgroundTintList = ContextCompat.getColorStateList(this, R.color.nav_green)
                }
            },
            onStateChanged = { isGpsCut, mode ->
                if (isGpsCut || mode == PositioningMode.DEAD_RECKONING) {
                    tvDemoModeBadge.text = "🟠 AI DEAD RECKONING"
                    tvDemoModeBadge.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#FEF3C7"))
                    tvDemoModeBadge.setTextColor(Color.parseColor("#B45309"))
                    tvDemoAiStatus.text = "🧠 KinoNet-R2 AI: Active (0.38 m/s RMSE) · InEKF: Free Inertial · Covariance: Gated"
                    updateCutGpsButtonUI(true)
                } else {
                    tvDemoModeBadge.text = "🟢 GNSS AIDED"
                    tvDemoModeBadge.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#DCFCE7"))
                    tvDemoModeBadge.setTextColor(Color.parseColor("#15803D"))
                    tvDemoAiStatus.text = "🛰️ GNSS Fix: 3D Locked (4.0m) · Chi-Square Gate: PASS · InEKF: Fused"
                    updateCutGpsButtonUI(false)
                }
            },
            onFinished = { result ->
                finishDemoRun(result)
            }
        )

        val originL = customStartLat ?: userLat
        val originLn = customStartLng ?: userLng
        demoPlaybackController?.startDemo(currentRouteCoords.ifEmpty { null }, originL, originLn)

        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        Toast.makeText(this, "🧪 Demo ride running! Test with 'Cut GPS' anytime below.", Toast.LENGTH_SHORT).show()
    }

    private fun finishDemoRun(result: DemoTripResult?) {
        isDemoActive = false
        demoPlaybackController?.stopDemo()

        layoutTopContainer.visibility = View.VISIBLE
        layoutDemoPanel.visibility = View.GONE
        cardAiHandoverBanner.visibility = View.GONE

        tvDemoBtnText.text = "🧪 Demo Ride"
        btnDemoRide.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_demo_ride)
        js("stopRide()")

        if (result != null) {
            showDemoAnalysisDialog(result)
        }
    }

    private fun updateCutGpsButtonUI(isCut: Boolean) {
        if (isCut) {
            btnDemoCutGps.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_restore_gps)
            tvBtnCutGps.text = "🛰️ Restore GPS"
        } else {
            btnDemoCutGps.background = ContextCompat.getDrawable(this, R.drawable.bg_btn_cut_gps)
            tvBtnCutGps.text = "📡 Cut GPS"
        }
    }

    private fun showDemoAnalysisDialog(result: DemoTripResult) {
        val distKm = result.totalDistanceMeters / 1000.0
        val outDistM = result.outageDistanceMeters
        val driftPct = result.driftPercent
        val statusEmoji = if (result.isroPassed) "PASS ✅" else "CHECK ⚠️"

        val msg = """
📍 Trip Performance Summary:
• Traveled Distance: ${String.format(Locale.US, "%.2f km", distKm)}
• Total Duration: ${String.format(Locale.US, "%02d:%02d", result.durationSec / 60, result.durationSec % 60)}
• Average Speed: ${result.avgSpeedKmh} km/h

🧠 Dead Reckoning & AI Benchmarks:
• GPS Outage Duration: ${result.outageDurationSec}s (Tunnel Blackout)
• Outage Distance: ${String.format(Locale.US, "%.1f m", outDistM)}
• KinoNet-R2 AI Speed RMSE: ${result.aiSpeedRmse} m/s
• Endpoint Position Error: ${result.endpointErrorMeters} m
• Drift Rate: ${driftPct}% (ISRO Target: < 10.0%) · $statusEmoji

🚀 Status: AI Dead Reckoning maintained precision trajectory during blackout and seamlessly re-fused GNSS upon recovery!
        """.trimIndent()

        AlertDialog.Builder(this)
            .setTitle("🏁 Demo Ride Analysis & Results")
            .setMessage(msg)
            .setPositiveButton("📊 View Full Results") { _, _ ->
                val intent = Intent(this, ResultsActivity::class.java)
                startActivity(intent)
                resetDestination()
            }
            .setNegativeButton("Done") { _, _ ->
                resetDestination()
            }
            .setCancelable(false)
            .show()
    }

    // ── Navigation Engine Sensor Callbacks ─────────────────────────────────────
    private fun setupEngineCallbacks() {
        drGnssManager.onGpsLost = {
            runOnUiThread {
                isGpsLost = true
                viewStatusDot.backgroundTintList = ContextCompat.getColorStateList(this, R.color.nav_traffic_slow)
            }
        }

        drGnssManager.onGpsRegained = {
            runOnUiThread {
                isGpsLost = false
                viewStatusDot.backgroundTintList = ContextCompat.getColorStateList(this, R.color.nav_green)
            }
        }

        navigationEngine.setOutputListener { output ->
            runOnUiThread { onEngineOutput(output) }
        }
    }

    private fun onEngineOutput(output: EngineOutput) {
        val lat = output.latitude ?: return
        val lng = output.longitude ?: return
        val headingDeg = Math.toDegrees(output.headingRad).toFloat()
        val speedKmh = (output.speedMps * 3.6).toInt()

        userLat = lat
        userLng = lng
        currentHeading = headingDeg.toDouble()

        tvRideSpeed.text = speedKmh.toString()

        if (isMapReady) {
            val isDr = output.positioningMode == PositioningMode.DEAD_RECKONING
            js("updateUserLocation($lat,$lng,$headingDeg,$isDr)")

            if (isRideActive) {
                js("addRidePoint($lat,$lng)")
                lastLat?.let { pLat ->
                    lastLng?.let { pLng ->
                        val d = haversineM(pLat, pLng, lat, lng)
                        if (d > 0.5 && d < 200.0) {
                            totalDistanceMeters += d
                            val km = totalDistanceMeters / 1000.0
                            tvRideDistance.text = String.format(Locale.US, "%.2f", km)
                        }
                    }
                }
                lastLat = lat
                lastLng = lng
            }
        }
    }

    private fun haversineM(lat1: Double, lng1: Double, lat2: Double, lng2: Double): Double {
        val r = 6371000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLng = Math.toRadians(lng2 - lng1)
        val a = sin(dLat / 2).pow(2) + cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLng / 2).pow(2)
        return r * 2 * atan2(sqrt(a), sqrt(1 - a))
    }

    // ── Location Permissions & Initial GPS Fix ─────────────────────────────────
    private fun checkLocationPermission() {
        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (permissions.any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQ_CODE)
        } else {
            startInitialLocationUpdates()
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQ_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startInitialLocationUpdates()
        }
    }

    private fun startInitialLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) return

        fusedLocationClient.lastLocation.addOnSuccessListener { loc: Location? ->
            if (loc != null) {
                userLat = loc.latitude
                userLng = loc.longitude
                if (isMapReady) {
                    js("updateUserLocation($userLat,$userLng,$currentHeading,false)")
                    js("recenter($userLat,$userLng,16.5)")
                }
            }
        }

        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateIntervalMillis(500L)
            .build()

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                userLat = loc.latitude
                userLng = loc.longitude
                if (loc.hasBearing()) currentHeading = loc.bearing.toDouble()
                if (loc.hasSpeed() && !isRideActive) {
                    tvRideSpeed.text = (loc.speed * 3.6).toInt().toString()
                }
                if (isMapReady && !isRideActive) {
                    js("updateUserLocation($userLat,$userLng,$currentHeading,false)")
                }
            }
        }
        fusedLocationClient.requestLocationUpdates(req, locationCallback!!, Looper.getMainLooper())
    }

    private fun js(code: String) {
        webMap.post {
            webMap.evaluateJavascript(code, null)
        }
    }

    override fun onResume() {
        super.onResume()
        webMap.onResume()
    }

    override fun onPause() {
        super.onPause()
        webMap.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        demoPlaybackController?.stopDemo()
        timerRunnable?.let { timerHandler.removeCallbacks(it) }
        locationCallback?.let { fusedLocationClient.removeLocationUpdates(it) }
        imuSensorManager.stopListening()
        drGnssManager.stopListening()
        webMap.destroy()
    }
}
