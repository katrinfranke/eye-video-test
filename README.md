# Eye-video analysis — online vs offline pupil tracking

Tools and reports for assessing the acquisition-time ("online") pupil tracker against a robust
offline detector, across two head-fixed setups (booths), and for checking whether a day-to-day
gaze bias is real behavior or a tracking artifact.

## Data layout

Each *animal/booth* folder holds timestamped *session* folders; each session is split into raw
grayscale AVI chunks `video_0.avi, …` (5 s each, 500 fps). Frame resolution varies per session.
OpenCV can't read these AVIs, so everything goes through **ffmpeg**.

- Booth 1: `/mnt/at-storageB1_I/EyeVideo/AT-B1NO1`
- Booth 2: `/mnt/at-storageB2_I/EyeVideo/AT-B2NO1`

## The two trackers

- **online** (`get_pupil_online`) — centroid of pixels in an intensity band `(low, high]` after a
  3×3 erosion. This is the acquisition-time method. `low = 0` always; `high` is set per booth by
  the experimenter (see thresholds).
- **offline / robust** (`detect_pupil_ellipse`) — dark-threshold → morphology → largest circular
  contour → ellipse center. Used as the reference "true" pupil center.

The dark-pixel count feeding the online centroid, `ndark`, is used as the pupil-size proxy; frames
with `ndark` below a per-booth **openness threshold** are treated as closed/occluded and excluded.

## Booth-specific thresholds

Set day-to-day by intuition; here fixed per booth from the pupil-size histograms:

| booth | `high` | openness `ndark >` | notes |
|---|---|---|---|
| 1 (AT-B1NO1) | 50 | 5000 | dark pupils (1st-pct intensity ≈ 40); high=60 over-includes iris |
| 2 (AT-B2NO1) | 40 | 11000 | larger/darker pupils |

The online tracker is **sensitive to `high`**: on booth 1 the online-vs-offline error is <1 px at
high=40–50 but ~15 px (≈3.9°) at high=60. Use the booth-appropriate value.

## Reports

- **[REPORT_accuracy.md](REPORT_accuracy.md)** (Report 1) — reliability of online vs offline
  tracking across both animals. Part A (tracking quality, pupil-size threshold) + Part B
  (offline-vs-online agreement, on-screen error). Built by `make_report_accuracy.py` →
  `results_accuracy.json`, `figures_acc/`.
  - Headline: at the correct per-booth threshold the online centroid matches the offline pupil to
    **< 1 px (≈ 0.15°) median, correlation ≈ 1.0** on both booths.
- **[REPORT.md](REPORT.md)** (Report 2) — booth-1 gaze-bias investigation. Is the day-to-day
  "centered vs biased" difference (from the enigma daily viewer) real gaze or an online-tracker
  artifact? Full **eye-anchored-frame** analysis: pupil position (horizontal/vertical) per condition,
  in both trackers. Built by `make_report.py` → `results.json`, `figures/`. Needs clicked landmarks
  (`eye_landmarks.json`).
  - Headline: the horizontal within-eye shift is real (present in both trackers, p ≈ 0.011), not a
    detection artifact.

## Code

- **`eyevideo.py`** — all shared helpers: session/frame IO, clip extraction, both pupil detectors,
  the eye-anchored frame, openness/closure, and every plot/analysis. Set `ev.ANIMAL_DIR`,
  `ev.OPEN_MIN`, and pass `high=` per booth.
- **`make_report_accuracy.py`** / **`make_report.py`** — the two report builders.
- **`single_session_inspection.ipynb`** — interactive: play a clip, static frames across a session,
  pupil-tracking examples, eye-frame pupil cloud for one session.
- **`across_sessions_eyeframe.ipynb`** — interactive version of the eye-frame condition comparison.

## Requirements

- `ffmpeg` / `ffprobe` on `PATH`
- Python: `numpy`, `opencv-python`, `matplotlib`, `scipy`, Jupyter; `ipympl` for landmark clicking.

## Reproduce

```python
# Report 1 (both booths, booth-specific thresholds inside the script):
python make_report_accuracy.py

# Report 2 (booth-1 gaze bias; needs eye_landmarks.json):
python make_report.py
```

Landmarks: `eye_landmarks.json` (booth-1 sessions). Tracking cache: `.track_cache/` keyed by
`(session, N, high)`, regenerated on demand (git-ignored).
