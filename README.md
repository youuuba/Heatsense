# HeatSense

Heat-strain estimation for runners from everyday wearable data: heart rate, pace, slope and
weather.

HeatSense reads your Garmin `.FIT` runs, cleans the heart-rate signal of real-world wrist
artifacts, and estimates your core temperature and heat strain minute by minute. It separates
"working harder" from "getting hot," and it personalizes from your own training history.

> **Status:** research prototype. Model and offline evaluation done; validation against a
> real core-temperature reference is next. **Not a medical device.** Zones are provisional.
> A low estimate never means you are safe: stop and cool down if you feel dizzy, confused or
> nauseous, and call emergency services for confusion or collapse.

## Quick start

```bash
pip install -e ".[dev]"
heatsense calibrate past_runs/*.fit --out profile.json    # learn your heart-rate-at-effort curve
heatsense analyze run.fit --profile profile.json --csv run.csv
heatsense demo        # simulated runner with known core temperature
heatsense evaluate    # HeatSense vs the reference baseline on simulated athletes
pytest
```

## How the model works

Once per minute:

1. **Physics.** Core temperature moves toward an equilibrium set by effort and heat stress,
   with a ~20-minute time constant. Effort is metabolic power from pace and slope
   (Minetti et al., 2002). Heat stress is WBGT from historical weather. When you stop,
   the equilibrium drops and you cool down.
2. **Heart rate is explained by several causes:**
   `HR = personal baseline + effort + heat (core temperature) + slow drift`, capped at
   your maximum heart rate.
3. **An unscented Kalman filter** compares predicted with measured heart rate and corrects
   the core-temperature *state*. Minutes flagged by the quality checks are skipped, and the
   uncertainty grows instead.
4. **Zones** (green / amber / red / check) come from the level, the 10-minute trend, heat
   stress and data quality. "Check" means the heart-rate signal was unreliable, so the
   app never shows false reassurance.

**Personal calibration** (`heatsense calibrate`) learns your heart-rate-at-effort curve
from the early, still-cool minutes of your past runs (minutes 4–12). It runs offline,
between runs, and the profile is frozen during a run.

## Results so far (simulated athletes)

The truth is known in simulation, so methods can be scored directly. The simulated
athletes deliberately use **different physiology equations** from the model (multiplicative
heat stress, asymmetric heating and cooling, hot-weather creep, non-linear heart-rate
response, dehydration drift, correlated noise, sensor artifacts), and five athletes differ in
how fast they heat up and in heart rate. Each athlete has 8 calibration runs and 30 different
test runs, and scoring starts after the first 10 minutes.

| Method | Bias (°C) | Error, RMSE (°C) | Minutes ≥ 38.8 °C caught | False alarms |
|---|---|---|---|---|
| Reference baseline (HR-only filter) | −0.14 | 0.49 | 42% | 6% |
| Physics only (effort + weather, no HR) | +0.06 | 0.36 | 73% | 4% |
| **HeatSense, calibrated** | +0.06 | **0.34** | **83%** | 6% |
| HeatSense, population defaults | −0.14 | 0.32 | 76% | 2% |

What this shows, and what it doesn't:

- Using effort and weather cuts the reference baseline's error by about a third. Heart rate
  adds a smaller refinement on top.
- **How fast an individual heats up cannot be learned from heart rate alone.** Heat gain,
  heat sensitivity and drift were not separable in calibration: consecutive minutes are
  highly correlated (r ≈ 0.85), so a handful of runs holds only tens of independent minutes.
  These stay at population values until runs with a core-temperature reference exist.
- Calibration is essential on real data: the population defaults were 17–29 bpm off on the
  author's runs; calibrated, the mean error is within −9 to +3 bpm.
- Simulation is not validation. Real accuracy needs a core-temperature reference
  (ingestible pill or CORE sensor). That is the next step.

## What's in here

| Module | What it does |
| --- | --- |
| `heatsense.io_fit` | Reads Garmin `.FIT` files (including CORE sensor `core_temperature`). GPS is dropped by default; minutes keep their real timeline across auto-pauses. |
| `heatsense.quality` | Flags bad heart rate: dropouts, out-of-range values, spikes, flatlines, cadence lock, and **soft lock** (the watch's smoothed heart rate snapping toward cadence at steady effort; found in real runs). |
| `heatsense.effort` | Grade from altitude, and metabolic power from pace and slope. |
| `heatsense.weather` | Historical weather from Open-Meteo (location rounded to ~1 km), heat index and approximate WBGT. Cached. |
| `heatsense.model` | The HeatSense physics, the filter, offline calibration and strain zones. |
| `heatsense.baseline` | Reference implementation of the published heart-rate-only filter (offline comparison only). |
| `heatsense.synthetic`, `heatsense.evaluate` | Simulated athletes and the head-to-head evaluation. |
| `heatsense.plot` | One-figure run summary. |

## Known limitations

- The physics coefficients are literature-informed starting values, not validated constants.
- The WBGT estimate assumes moderate sun and light wind; treadmill runs assume indoor
  WBGT 18 °C.
- Watch treadmill pace is estimated from the wrist and can be off by ~10%, which skews
  effort. A foot pod helps.
- It doesn't work for people whose heart rate doesn't reflect heat (e.g. beta-blockers,
  pacemakers).

## Prior work and patent notice

HeatSense builds on published research on estimating core temperature from heart rate:

> Buller MJ, Tharion WJ, Cheuvront SN, et al. *Estimation of human core temperature from
> sequential heart rate observations.* Physiological Measurement 34(7):781–798, 2013.

> Laxminarayan S, Rakesh V, et al. *Individualized estimation of human core body temperature
> using noninvasive measurements.* Journal of Applied Physiology, 2018.

The heart-rate-only method is patented by the US Army (**US 10,702,165**). In this project it
appears **only as a clearly labeled offline reference baseline** (`heatsense/baseline.py`) for
comparing accuracy on recorded data. It is not shipped in any watch or mobile app, and
HeatSense has no heart-rate-only mode.

The US Army also holds patents on individualized estimation from physiological and
environmental data (**US 11,517,203**, **US 11,540,723**). HeatSense is designed differently:
no skin-temperature input, personal parameters are fitted offline between runs and stay fixed
during a run, and the in-run filter updates only the temperature state, with no
error-triggered parameter changes.

This is an engineering description, not legal advice.

## License

Apache License 2.0, see `LICENSE` and `NOTICE`. The license covers this code only and
grants no rights under any third-party patent.
