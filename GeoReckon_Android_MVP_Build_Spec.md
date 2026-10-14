# GeoReckon --- Android MVP Build Specification

**Project:** GeoReckon / SIH26168\
**Purpose:** Offline-first Android demonstrator for physics-first
vehicle dead reckoning with learned speed assistance\
**Target:** MVP / hackathon demonstration\
**Platform:** Android\
**Primary UI:** Kotlin + Jetpack Compose\
**Status:** Build specification derived from the GeoReckon revised
architecture and the current `georeckon_pipeline.py`

------------------------------------------------------------------------

## 1. MVP Objective

Build an Android application that demonstrates the following complete
flow:

``` text
GNSS available
      ↓
GNSS-aided navigation
      ↓
GNSS outage / simulated blackout
      ↓
IMU + physics propagation
      +
learned forward-speed estimate
      ↓
Dead reckoning
      ↓
GNSS returns
      ↓
Reacquisition / correction
      ↓
GNSS-aided navigation
```

The MVP is **not** a production navigation product and must not claim
production-level accuracy.

The first demonstrator must prove that the system can:

1.  initialize from a known GNSS state,
2.  display a vehicle trajectory,
3.  enter a GNSS-denied state,
4.  continue estimating position during the outage,
5.  expose the active positioning mode and uncertainty,
6.  recover when GNSS returns,
7.  show the resulting trajectory and measured/simulated error.

The revised architecture explicitly defines the Android demonstration,
reusable edge engine, preliminary trained model, and reproducible
benchmark plots as deliverables. Numerical claims must be labelled
**Measured**, **Simulation**, or **Target**.

------------------------------------------------------------------------

# 2. Product Scope

## 2.1 MVP features

### Required

-   Offline-first operation
-   Android app
-   Jetpack Compose UI
-   Offline map display
-   Recorded-trip replay
-   GNSS-aided mode
-   Dead-reckoning mode
-   GNSS recovery mode
-   Vehicle position marker
-   Estimated trajectory
-   GNSS/reference trajectory for post-run comparison
-   Speed display
-   Positioning mode indicator
-   Health indicator
-   Uncertainty indicator
-   Demo controls
-   Trip result screen
-   Local persistence
-   Local ML model/inference architecture
-   Local experiment/session logs

### Optional if time permits

-   Live smartphone IMU recording
-   Live GNSS recording
-   ONNX Runtime inference on Android
-   Sensor diagnostics screen
-   Trip export
-   Simple offline road graph
-   Soft road constraint

### Explicitly out of MVP scope

-   Cloud backend
-   Firebase in the live positioning loop
-   User accounts
-   Cloud inference
-   Worldwide offline maps
-   Full production road matching
-   Lane-level localization
-   Parking-floor localization
-   External IMU hardware support
-   Fully optimized C++ engine
-   Guaranteed bounded navigation error
-   Production-grade turn-by-turn navigation

The revised architecture states that a cloud service is not required in
the live positioning loop and treats offline map matching as a
later/final-solution capability.

------------------------------------------------------------------------

# 3. MVP Architecture

``` text
                           GEORECKON ANDROID MVP
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
           REPLAY / DEMO MODE                         LIVE MODE
                 │                                         │
        Recorded IMU/GNSS data                       Android sensors
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ↓
                              Input Normalizer
                                      ↓
                             Physics Estimator
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                       IMU                  GNSS when valid
                         │                         │
                         └────────────┬────────────┘
                                      ↓
                              Learned Speed Model
                                  KinoNet-R2
                                      │
                              speed + variance
                                      ↓
                             Fusion / Constraints
                                      ↓
                        ┌─────────────┼─────────────┐
                        ↓             ↓             ↓
                     Position       Speed        Health
                        │
                        ↓
                  Offline Map
                        │
                        ↓
                  Compose UI
```

The revised system architecture defines:

-   timestamped IMU + quality checks,
-   physics propagation,
-   learned forward speed + uncertainty,
-   GNSS/road evidence,
-   fusion and constraint management,
-   position + uncertainty + mode.

------------------------------------------------------------------------

# 4. Important Architecture Decision

## MVP implementation strategy

Do **not** begin by implementing the complete proposed C++/Eigen + JNI
architecture.

Use this development order:

``` text
Python research pipeline
        ↓
validate one end-to-end dataset sequence
        ↓
produce deterministic replay output
        ↓
build Android replay demonstrator
        ↓
export/test ML model
        ↓
connect local inference
        ↓
add live Android sensors
        ↓
refactor reusable estimator into C++/Eigen if required
```

The proposed final architecture uses C++ + Eigen + CMake for the shared
engine and NDK/JNI for Android integration. That is the target
architecture, not a prerequisite for the first demonstrator.

------------------------------------------------------------------------

# 5. Offline-First Design

The app must continue operating without internet access during a
navigation session.

## Local components

``` text
Android
├── Offline map resources
├── Local ML model
├── Preprocessing configuration
├── Physics estimator
├── Room / SQLite database
├── Demo datasets
└── Session logs
```

## No network dependency

The live positioning loop must not require:

-   Firebase
-   REST API
-   cloud server
-   cloud ML inference
-   internet connectivity

Internet may be used outside the navigation loop for:

-   downloading map resources,
-   updating models,
-   importing datasets,
-   developer diagnostics.

------------------------------------------------------------------------

# 6. Android Technology Stack

  --------------------------------------------------------------------------------
  Layer                   MVP Technology          Responsibility
  ----------------------- ----------------------- --------------------------------
  UI                      Kotlin + Jetpack        Screens, controls, status
                          Compose                 

  Sensor acquisition      `SensorManager`         Accelerometer/gyroscope

  Location                `LocationManager` /     GNSS
                          Android location APIs   

  Local storage           Room / SQLite           Trips, sessions, logs

  Map                     MapLibre Native         Offline map rendering

  ML                      ONNX Runtime            Local model inference

  Research/training       Python, NumPy, pandas,  Training and evaluation
                          SciPy, PyTorch          

  Future shared engine    C++ + Eigen + CMake     Propagation/fusion/constraints

  Android native bridge   NDK + JNI               Future engine integration

  Dataset format          CSV / Parquet           Research/replay data

  App replay format       JSON initially          Simple deterministic MVP replay
  --------------------------------------------------------------------------------

------------------------------------------------------------------------

# 7. Android Project Structure

Recommended package structure:

``` text
app/
└── src/main/java/com/georeckon/
    ├── MainActivity.kt
    │
    ├── ui/
    │   ├── navigation/
    │   │   ├── AppNavGraph.kt
    │   │   └── Routes.kt
    │   │
    │   ├── home/
    │   │   ├── HomeScreen.kt
    │   │   └── HomeViewModel.kt
    │   │
    │   ├── navigation/
    │   │   ├── NavigationScreen.kt
    │   │   └── NavigationViewModel.kt
    │   │
    │   ├── diagnostics/
    │   │   └── DiagnosticsScreen.kt
    │   │
    │   └── results/
    │       └── ResultsScreen.kt
    │
    ├── data/
    │   ├── local/
    │   │   ├── GeoReckonDatabase.kt
    │   │   ├── TripDao.kt
    │   │   └── TripEntity.kt
    │   │
    │   ├── replay/
    │   │   ├── DemoTrip.kt
    │   │   └── DemoTripRepository.kt
    │   │
    │   └── model/
    │       ├── ImuSample.kt
    │       ├── GnssSample.kt
    │       ├── NavigationState.kt
    │       └── EngineOutput.kt
    │
    ├── sensors/
    │   ├── ImuSensorManager.kt
    │   └── GnssManager.kt
    │
    ├── engine/
    │   ├── NavigationEngine.kt
    │   ├── PhysicsEstimator.kt
    │   ├── FusionManager.kt
    │   └── EngineConfig.kt
    │
    ├── ml/
    │   ├── KinoNetRunner.kt
    │   ├── ModelPreprocessor.kt
    │   └── ModelConfig.kt
    │
    ├── map/
    │   ├── OfflineMapController.kt
    │   └── TrajectoryRenderer.kt
    │
    └── session/
        ├── SessionManager.kt
        └── DemoController.kt
```

------------------------------------------------------------------------

# 8. Core Data Contracts

The revised architecture requires timestamps to be preserved from the
sensor source. Callback arrival time must not replace sensor time.

## 8.1 IMU sample

``` kotlin
data class ImuSample(
    val timestampNs: Long,
    val accelX: Double,
    val accelY: Double,
    val accelZ: Double,
    val gyroX: Double,
    val gyroY: Double,
    val gyroZ: Double,
    val frameId: String,
    val qualityFlags: Int
)
```

Units:

``` text
Acceleration: m/s²
Angular rate: rad/s
Timestamp: sensor time
```

------------------------------------------------------------------------

## 8.2 GNSS sample

``` kotlin
data class GnssSample(
    val timestampNs: Long,
    val latitude: Double,
    val longitude: Double,
    val speedMps: Double?,
    val accuracyM: Double?,
    val valid: Boolean
)
```

------------------------------------------------------------------------

## 8.3 Engine output

``` kotlin
data class EngineOutput(
    val timestampNs: Long,
    val latitude: Double?,
    val longitude: Double?,
    val localX: Double,
    val localY: Double,
    val localZ: Double,
    val velocityX: Double,
    val velocityY: Double,
    val velocityZ: Double,
    val headingRad: Double,
    val covariance: DoubleArray,
    val positioningMode: PositioningMode,
    val health: EngineHealth,
    val alignmentHealthy: Boolean,
    val gnssAccepted: Boolean,
    val aiAccepted: Boolean,
    val modelVersion: String
)
```

------------------------------------------------------------------------

# 9. Positioning Modes

The app must distinguish positioning mode from sensor/system health.

## Positioning mode

``` kotlin
enum class PositioningMode {
    INITIALIZING,
    GNSS_AIDED,
    DEAD_RECKONING,
    REACQUIRING
}
```

### INITIALIZING

Establish:

-   initial position,
-   velocity,
-   heading where observable,
-   tilt,
-   sensor bias estimate,
-   initial uncertainty.

Gravity can provide tilt but cannot independently determine yaw.

### GNSS_AIDED

-   IMU continuously propagates state.
-   Valid GNSS updates are fused.
-   GNSS fixes are checked for quality/consistency before acceptance.

### DEAD_RECKONING

-   GNSS is unavailable or intentionally disabled for the outage test.
-   IMU propagation continues.
-   Valid local corrections may continue.
-   Reference GNSS must not leak into the estimator during evaluation.

### REACQUIRING

-   Returning GNSS fixes are checked over a consistency window.
-   Timestamp delays are handled.
-   The state is corrected with appropriate uncertainty.
-   The first returning fix must not simply snap the trajectory.

------------------------------------------------------------------------

# 10. Health State

``` kotlin
enum class EngineHealth {
    NORMAL,
    ADAPTIVE,
    DEGRADED,
    RECOVERING
}
```

Examples:

``` text
GNSS available + good sensors
→ GNSS_AIDED + NORMAL

GNSS unavailable + usable IMU
→ DEAD_RECKONING + ADAPTIVE

Sensor saturation / bad alignment
→ DEAD_RECKONING + DEGRADED

GNSS returning
→ REACQUIRING + RECOVERING
```

------------------------------------------------------------------------

# 11. Physics Estimator

The physics branch is the foundation.

State:

``` text
R  = orientation
v  = velocity
p  = position
ba = accelerometer bias
bg = gyroscope bias
```

Conceptual propagation:

``` text
R_next = R Exp([ω - bg]x dt)

a_world = R(f - ba) + g

v_next = v + a_world dt

p_next = p + v dt + 0.5 a_world dt²
```

These are illustrative discrete equations. Production integration must
correctly handle changing orientation, numerical integration and
evolving bias uncertainty.

## Critical requirements

-   Use actual sensor timestamps.
-   Calculate `dt` from timestamps.
-   Explicitly define coordinate frames.
-   Preserve raw IMU data.
-   Track sensor quality.
-   Do not assume a fixed 0.01 s timestep.
-   Do not freeze bias forever.
-   Do not use future samples.

------------------------------------------------------------------------

# 12. Phone-to-Vehicle Alignment

The phone coordinate frame is not automatically the vehicle coordinate
frame.

The system must establish:

``` text
Phone frame
     ↓
Phone → vehicle transform
     ↓
Vehicle frame
     ↓
World/local frame
```

Initial MVP:

1.  Detect a valid stationary period.
2.  Estimate roll/pitch from gravity.
3.  Estimate forward direction from verified motion.
4.  Establish phone-to-vehicle alignment.
5.  Store alignment quality.
6.  Reinitialize if the phone is moved significantly.

If alignment becomes unreliable:

``` text
alignmentHealthy = false
```

and affected corrections should be weakened or disabled.

------------------------------------------------------------------------

# 13. Sensor Conditioning

The revised architecture requires causal, online-safe processing.

Do not use an offline-only filtering method in the online Android path
if it requires future samples.

The current Python research pipeline uses SciPy filtering and therefore
must be treated as a research/offline pipeline until its online
preprocessing is replaced or verified.

MVP requirements:

-   timestamp validation,
-   duplicate sample detection,
-   missing sample detection,
-   saturation flags,
-   unit conversion,
-   robust outlier detection,
-   causal filtering where needed,
-   quality-dependent noise.

Keep a raw-data path because aggressive low-pass filtering can remove
features needed by the neural model.

------------------------------------------------------------------------

# 14. KinoNet-R2 / Learned Speed

The current Python pipeline defines KinoNet-R2 as a model that consumes
a recent IMU window and predicts:

``` text
forward speed μ
prediction uncertainty / variance
```

The revised architecture recommends the smallest possible learned
contribution:

``` text
IMU window
    ↓
Causal CNN / CNN-GRU
    ↓
signed forward speed
+
positive variance
```

Do not initially predict latitude/longitude directly.

## Input

Initial model contract:

``` text
6 channels:

Accel X
Accel Y
Accel Z
Gyro X
Gyro Y
Gyro Z
```

The exact sample rate and window length must be taken from the validated
training configuration rather than assumed.

The current Python prototype uses a 2-second / 20-sample window at its
configured 10 Hz dataset rate; this must be verified against the actual
dataset before deployment.

## Output

``` text
timestamp
forwardSpeedMps
variance
inputValid
modelVersion
```

------------------------------------------------------------------------

# 15. AI Uncertainty

Do not treat a neural-network confidence value as a probability of
correctness.

The model should produce a physically meaningful variance.

Example:

``` text
speed = 12.4 m/s
variance = 0.25 (m/s)²
```

The fusion engine converts this into a bounded measurement covariance.

Conceptually:

``` text
Low uncertainty
      ↓
AI measurement receives more weight

High uncertainty
      ↓
AI measurement receives less weight
```

If the model is invalid or unreliable:

``` text
AI rejected
     ↓
physics continues
     ↓
uncertainty increases
```

------------------------------------------------------------------------

# 16. Fusion

For the first implementation, use a tested error-state Kalman-filter
approach if a complete InEKF implementation is not yet validated.

The revised architecture specifically recommends:

> Use a tested error-state Kalman filter initially; implement the
> proposed InEKF only with documented error definitions, Jacobians and
> covariance reset.

The current Python prototype contains an `InvariantEKF` class, but the
Android MVP must not assume that its implementation is
production-validated merely because the class exists.

------------------------------------------------------------------------

# 17. Vehicle Constraints

## Non-holonomic constraint

For a normal vehicle:

``` text
lateral velocity ≈ 0
vertical velocity ≈ 0
```

Use this as a **soft measurement**, not an unconditional clamp.

Weaken the constraint when:

-   slip is detected,
-   vehicle is banking,
-   suspension movement is significant,
-   sensor quality is poor,
-   the supported vehicle class violates the assumption.

------------------------------------------------------------------------

## ZUPT

When a robust stop detector confirms that the vehicle is stationary:

``` text
velocity ≈ 0
```

Apply a zero-velocity update.

Do not infer "stopped" from low IMU variance alone because smooth
cruising can also have low variance.

------------------------------------------------------------------------

# 18. GNSS Recovery

The demo should deliberately show:

``` text
GNSS
  ↓
BLACKOUT
  ↓
DEAD RECKONING
  ↓
GNSS RETURN
  ↓
REACQUIRING
  ↓
GNSS-AIDED
```

Recovery algorithm requirements:

1.  Detect returning GNSS.
2.  Validate timestamp.
3.  Check fix quality.
4.  Check consistency with predicted state.
5.  Reject obviously inconsistent fixes.
6.  Apply correction using uncertainty.
7.  Log whether the fix was accepted or rejected.
8.  Return to GNSS-aided mode.

Do not simply snap the marker to the first returning GPS point.

------------------------------------------------------------------------

# 19. Offline Demo Mode

This is the primary MVP demonstration path.

## Demo data

Bundle a deterministic recorded trip:

``` text
assets/
└── demo/
    ├── trip.json
    ├── metadata.json
    └── trajectory.json
```

Example metadata:

``` json
{
  "tripId": "demo_001",
  "sampleRateHz": 10,
  "outageStartSec": 30,
  "outageEndSec": 60,
  "modelVersion": "kinonet-r2-001",
  "preprocessingVersion": "prep-001",
  "dataType": "SIMULATION"
}
```

The exact values must be generated from an actual validated run rather
than invented.

------------------------------------------------------------------------

# 20. Demo Controller

The demo controller is intentionally deterministic.

``` text
START
  ↓
INITIALIZING
  ↓
GNSS_AIDED
  ↓
OUTAGE START
  ↓
DEAD_RECKONING
  ↓
OUTAGE END
  ↓
REACQUIRING
  ↓
GNSS_AIDED
  ↓
RESULTS
```

Controls:

``` text
[ Start Demo ]

[ Pause ]

[ Restart ]

[ Simulate GNSS Loss ]

[ Restore GNSS ]
```

For the official presentation, prefer a predefined outage timeline so
the demonstration is reproducible.

------------------------------------------------------------------------

# 21. Main Android Screens

## 21.1 Home

``` text
GeoReckon

Offline-first GNSS-denied navigation

[ Start Demo ]

[ Live Sensor Test ]

[ Previous Sessions ]
```

------------------------------------------------------------------------

## 21.2 Navigation

Main screen:

``` text
┌──────────────────────────────┐
│ GeoReckon          12:32     │
├──────────────────────────────┤
│                              │
│             MAP              │
│                              │
│          ───────             │
│             🚗               │
│                              │
│                              │
├──────────────────────────────┤
│ Positioning                  │
│ GNSS-AIDED                   │
│                              │
│ Health                       │
│ NORMAL                       │
│                              │
│ Speed          42 km/h       │
│ Uncertainty    8.2 m         │
│                              │
│ [ End Session ]              │
└──────────────────────────────┘
```

During outage:

``` text
GNSS            LOST
Positioning     DEAD RECKONING
Health          ADAPTIVE
Speed           42 km/h
Uncertainty     14.8 m
```

------------------------------------------------------------------------

# 22. Diagnostics Screen

For judges/developers:

``` text
SYSTEM DIAGNOSTICS

IMU
Accelerometer      10 Hz / actual rate
Gyroscope          actual rate

TIMESTAMP
Last sensor dt     0.0102 s

ALIGNMENT
Status             HEALTHY

AI
Model              KinoNet-R2
Speed              11.7 m/s
Variance           0.32
Accepted           YES

FILTER
Positioning        DEAD_RECKONING
GNSS update        REJECTED / UNAVAILABLE
NHC                ACTIVE
ZUPT               INACTIVE

QUALITY
Dropped samples    0
Saturation         NONE
```

This screen is optional for end users but valuable during the hackathon.

------------------------------------------------------------------------

# 23. Results Screen

Show the actual evidence:

``` text
TRIP RESULT

Mode
GNSS → DR → GNSS

Outage duration
640 m / 30 s

Endpoint error
XX m

RMS position error
XX m

Maximum error
XX m

Endpoint drift
XX %

AI speed error
XX m/s

[ View Trajectory ]
```

Every metric must be labelled:

``` text
Measured
Simulation
Target
```

Do not present simulated values as device measurements.

------------------------------------------------------------------------

# 24. Trajectory Visualization

The results screen should show at least:

``` text
Reference / GNSS trajectory
GeoReckon estimated trajectory
```

During the outage:

``` text
Reference
───────────────

GeoReckon
──────────────╮
              ╰────
```

After GNSS recovery:

``` text
Reference
────────────────────

GeoReckon
──────────────╮
              ╰────────────
```

The reference path is for scoring/display only. It must never be fed to
the estimator during the outage.

------------------------------------------------------------------------

# 25. Offline Map

Use MapLibre Native for the map display.

The map layer is responsible for:

-   rendering downloaded resources,
-   showing vehicle position,
-   drawing reference trajectory,
-   drawing estimated trajectory,
-   optionally showing road geometry.

Map rendering is separate from the navigation estimator.

For the MVP:

``` text
MapLibre
    +
offline map resources
    +
trajectory overlays
```

Do not build full HMM road matching unless the core MVP is already
working.

------------------------------------------------------------------------

# 26. Local Persistence

Use Room/SQLite.

Suggested tables:

## sessions

``` text
id
startTime
endTime
mode
modelVersion
preprocessingVersion
resultType
```

## navigation_samples

``` text
sessionId
timestamp
latitude
longitude
localX
localY
speed
uncertainty
mode
health
```

## diagnostics

``` text
sessionId
timestamp
droppedSamples
sensorQuality
alignmentHealth
aiAccepted
gnssAccepted
```

The database is local and remains available offline.

------------------------------------------------------------------------

# 27. Python Pipeline Role

The current `georeckon_pipeline.py` is the research/training/evaluation
pipeline.

It defines five phases:

``` text
Phase 1
Signal cleaning

Phase 2
Phone-to-vehicle alignment

Phase 3
KinoNet-R2 speed prediction

Phase 4
Physics filter / InEKF

Phase 5
Validation
```

The current script's main function runs these phases sequentially and
reports a final drift percentage.

The Python pipeline should remain the reference implementation during
MVP development.

------------------------------------------------------------------------

# 28. Python → Android Artifact Pipeline

``` text
IO-VNBD / validated dataset
             ↓
      Python pipeline
             ↓
     ┌───────┼─────────┐
     ↓       ↓         ↓
   model   replay    metrics
    .pth    data      plots
     │
     ↓
   ONNX
     │
     ↓
 Android local inference
```

Generate a versioned package:

``` text
georeckon_artifacts/
├── model.onnx
├── preprocessing.json
├── model_metadata.json
├── demo_trip.json
├── benchmark.json
└── README.md
```

------------------------------------------------------------------------

# 29. Model Export

Export KinoNet-R2 to ONNX early.

Before using it in Android:

1.  Save a fixed input tensor in Python.
2.  Run PyTorch inference.
3.  Export ONNX.
4.  Run ONNX inference.
5.  Compare outputs.
6.  Establish an acceptable numerical tolerance.
7.  Only then integrate into Android.

Do not quantize until floating-point inference has been verified.

------------------------------------------------------------------------

# 30. Replay vs Live Sensor Mode

## Replay mode

``` text
Recorded data
    ↓
Android
    ↓
Engine
    ↓
Map
```

Purpose:

-   deterministic judging demo,
-   reproducibility,
-   debugging,
-   algorithm comparison.

## Live mode

``` text
SensorManager
    ↓
timestamped IMU
    ↓
local engine
    ↓
Map
```

Purpose:

-   demonstrate actual phone integration,
-   collect new data,
-   benchmark on-device processing.

Replay mode is the guaranteed MVP path.

------------------------------------------------------------------------

# 31. MVP Development Milestones

## Milestone 1 --- Python baseline

**Goal:** One dataset sequence executes end-to-end.

Deliver:

-   trajectory,
-   GNSS outage interval,
-   DR trajectory,
-   speed prediction,
-   uncertainty,
-   error metrics.

------------------------------------------------------------------------

## Milestone 2 --- Deterministic replay artifact

Create:

``` text
demo_trip.json
```

with:

-   timestamp,
-   reference position,
-   GNSS availability,
-   estimated position,
-   speed,
-   uncertainty,
-   mode.

------------------------------------------------------------------------

## Milestone 3 --- Android shell

Build:

-   Home,
-   Navigation,
-   Results.

No real engine yet.

------------------------------------------------------------------------

## Milestone 4 --- Offline map

Add MapLibre and local map resources.

Render the replay trajectory.

------------------------------------------------------------------------

## Milestone 5 --- GNSS blackout visualization

Implement:

``` text
GNSS_AIDED
     ↓
DEAD_RECKONING
     ↓
REACQUIRING
     ↓
GNSS_AIDED
```

------------------------------------------------------------------------

## Milestone 6 --- Local ML inference

Export KinoNet-R2 to ONNX.

Integrate ONNX Runtime.

Verify Python vs Android outputs.

------------------------------------------------------------------------

## Milestone 7 --- Local physics engine

Implement or connect:

-   orientation,
-   bias,
-   velocity,
-   position,
-   covariance,
-   NHC,
-   ZUPT,
-   GNSS update.

------------------------------------------------------------------------

## Milestone 8 --- Live sensors

Add:

-   accelerometer,
-   gyroscope,
-   GNSS,
-   sensor timestamps,
-   quality flags.

------------------------------------------------------------------------

## Milestone 9 --- C++ engine

If required for the final architecture:

``` text
C++ + Eigen
      ↓
CMake
      ↓
NDK/JNI
      ↓
Kotlin
```

------------------------------------------------------------------------

## Milestone 10 --- Benchmark

Measure:

-   Android output frequency,
-   processing latency,
-   dropped samples,
-   memory,
-   energy,
-   model inference latency.

A desktop replay speed must not be presented as phone performance.

------------------------------------------------------------------------

# 32. Acceptance Criteria

## Functional

-   [ ] App starts without internet.
-   [ ] Demo loads entirely from local resources.
-   [ ] Offline map renders.
-   [ ] Demo begins in GNSS-aided mode.
-   [ ] GNSS blackout is detected/simulated.
-   [ ] App switches to dead reckoning.
-   [ ] Estimated position continues during blackout.
-   [ ] Uncertainty changes during outage.
-   [ ] GNSS recovery is represented separately.
-   [ ] Returning GNSS is validated before correction.
-   [ ] Final trajectory can be reviewed offline.
-   [ ] Session results are stored locally.

## Data integrity

-   [ ] Sensor timestamps are preserved.
-   [ ] `dt` is derived from timestamps.
-   [ ] Units are explicit.
-   [ ] Coordinate frames are explicit.
-   [ ] Missing/duplicate samples are detected.
-   [ ] Reference trajectory never enters the estimator during outage
    evaluation.

## ML

-   [ ] Model version is stored.
-   [ ] Preprocessing version is stored.
-   [ ] Python/ONNX output parity is tested.
-   [ ] Invalid model output causes fallback rather than uncontrolled
    correction.

## Demonstration

-   [ ] Demo is deterministic.
-   [ ] Demo works without internet.
-   [ ] Demo does not depend on driving conditions.
-   [ ] Results clearly label Simulation / Measured / Target.
-   [ ] Trajectory comparison is visible.

------------------------------------------------------------------------

# 33. Evaluation Plan

Use the staged experiments defined by the revised architecture.

### A --- Inertial baseline

Measure drift from basic inertial propagation.

### B --- Physics + constraints

Measure the contribution of:

-   alignment,
-   bias handling,
-   NHC,
-   valid ZUPT.

### C --- Physics + learned speed

Add learned speed with conservative fixed covariance.

### D --- Learned uncertainty

Allow model uncertainty to affect weighting.

### E --- Independent map

Add map contribution separately.

### F --- Continuous GNSS fusion

Evaluate GNSS-aided operation, outage and recovery.

For every outage record:

-   outage duration,
-   reference path length,
-   initialization method,
-   start/end time,
-   endpoint error,
-   maximum error,
-   RMS position error,
-   error at 50 m where applicable,
-   along-track error,
-   cross-track error,
-   heading error where reliable,
-   prediction uncertainty,
-   failure flags.

Endpoint drift:

``` text
Endpoint Drift (%) =
100 × endpoint position error / distance travelled during outage
```

For near-zero-distance segments, report absolute error instead of an
unstable percentage.

------------------------------------------------------------------------

# 34. Failure Handling

## Phone moved

``` text
alignment unhealthy
      ↓
weaken affected updates
      ↓
reinitialize alignment when observable
```

## Rough road / sensor saturation

``` text
quality flag
      ↓
increase measurement noise
      or
reject unsuitable correction
```

## No reliable AI evidence

``` text
reject AI update
      ↓
continue physics
      ↓
increase uncertainty
```

## Bad GNSS return

``` text
GNSS fix
   ↓
consistency check
   ↓
reject if inconsistent
   ↓
log rejection reason
```

## Stop/cruise ambiguity

Never set velocity to zero simply because acceleration variance is low.

------------------------------------------------------------------------

# 35. Demo Script

The presentation should follow this sequence:

### 1. Start

``` text
GeoReckon
Offline-first navigation
```

### 2. Initialize

``` text
Position: initialized
GNSS: available
Mode: GNSS-AIDED
```

### 3. Navigate

Show vehicle moving on the offline map.

### 4. GNSS blackout

Trigger the predetermined outage.

UI:

``` text
GNSS: LOST
Mode: DEAD RECKONING
Health: ADAPTIVE
```

### 5. Continue

Show:

-   vehicle movement,
-   estimated speed,
-   uncertainty,
-   estimated trajectory.

### 6. GNSS recovery

``` text
GNSS: RETURNED
Mode: REACQUIRING
```

Then:

``` text
Mode: GNSS-AIDED
```

### 7. Results

Display:

-   outage duration,
-   endpoint error,
-   RMS error,
-   maximum error,
-   drift,
-   trajectory comparison.

------------------------------------------------------------------------

# 36. What the MVP Is Actually Proving

The MVP is not proving that a smartphone can guarantee navigation
accuracy in every GNSS-denied environment.

It is proving the engineering loop:

``` text
IMU
 ↓
physics propagation
 ↓
learned speed evidence
 ↓
uncertainty-aware fusion
 ↓
GNSS outage
 ↓
dead reckoning
 ↓
GNSS recovery
 ↓
correction
 ↓
measured result
```

That is the correct scope for the first demonstrator.

------------------------------------------------------------------------

# 37. Critical Technical Notes

## Do not assume 100 Hz

The revised architecture specifically warns that IO-VNBD smartphone
recordings include 10 Hz data and says not to assume 100 Hz. The Android
system should process sensor events at their actual available rate and
target approximately 10 Hz navigation output.

## Do not use future samples

The learned branch must be causal during online evaluation.

## Do not use callback arrival time

Use the sensor measurement timestamp.

## Do not feed reference GNSS during blackout

Reference trajectories are labels/scoring data, not estimator inputs.

## Do not equate uncertainty with accuracy probability

Model variance is a measurement quantity used for weighting.

## Do not claim the current Python pipeline is production-ready

The revised PDF explicitly states that the proposed architecture and
performance remain to be verified.

------------------------------------------------------------------------

# 38. Recommended Repository

``` text
GeoReckon/
│
├── android/
│   ├── app/
│   └── README.md
│
├── python/
│   ├── georeckon_pipeline.py
│   ├── training/
│   ├── evaluation/
│   └── export/
│
├── engine/
│   └── future-cpp/
│
├── models/
│   ├── kinonet_r2.onnx
│   └── metadata.json
│
├── datasets/
│   └── manifests/
│
├── demo/
│   ├── trip.json
│   └── metadata.json
│
├── benchmarks/
│   ├── plots/
│   └── metrics/
│
└── docs/
    └── MVP_BUILD.md
```

------------------------------------------------------------------------

# 39. Final Build Order

Do this in exactly this order:

``` text
01. Run current Python pipeline
        ↓
02. Audit actual output and failures
        ↓
03. Select one reproducible dataset sequence
        ↓
04. Produce physics baseline
        ↓
05. Produce learned-speed result
        ↓
06. Produce outage trajectory + metrics
        ↓
07. Export deterministic demo JSON
        ↓
08. Create Android Compose project
        ↓
09. Add offline MapLibre display
        ↓
10. Replay demo trajectory
        ↓
11. Add GNSS/DR/recovery UI states
        ↓
12. Add Room session storage
        ↓
13. Export KinoNet-R2 to ONNX
        ↓
14. Verify ONNX/Python parity
        ↓
15. Add local ONNX inference
        ↓
16. Add physics estimator
        ↓
17. Add Android SensorManager
        ↓
18. Add live GNSS
        ↓
19. Add live sensor mode
        ↓
20. Benchmark the phone
        ↓
21. Add C++/JNI only where justified
        ↓
22. Package deterministic offline demo
```

------------------------------------------------------------------------

# 40. MVP Definition of Done

The GeoReckon Android MVP is complete when a judge can take the APK,
turn off internet access, open the application, start a bundled
demonstration, observe:

``` text
GNSS-AIDED
     ↓
GNSS LOST
     ↓
DEAD RECKONING
     ↓
GNSS RETURN
     ↓
REACQUIRING
     ↓
GNSS-AIDED
```

on an offline map, and then inspect the resulting trajectory and clearly
labelled evaluation metrics.

The live smartphone sensor path is an additional milestone, not a
prerequisite for the deterministic demonstration.

------------------------------------------------------------------------

## Source Basis

This specification is based on:

1.  `GeoReckon_Revised_Architecture_and_Implementation_Plan.pdf` ---
    revised engineering architecture, Android stack, interfaces, GNSS
    recovery, offline map strategy, evaluation plan and execution order.
2.  `georeckon_pipeline.py` --- current Python
    research/training/inference prototype containing the five-phase
    GeoReckon-R2 pipeline, KinoNet-R2 and the current physics-filter
    implementation.

The revised PDF explicitly states that the architecture is a proposed
engineering baseline and that implementation/performance still need
verification. Therefore this document treats the Python implementation
as the current prototype and the revised PDF as the intended
architecture, rather than representing either as already validated
production software.
