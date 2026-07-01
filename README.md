# Eye-video tools

Utilities for inspecting head-fixed eye-tracking recordings and checking whether apparent
day-to-day gaze bias is real or an artifact of the online pupil tracker.

## Data layout

An *animal* folder holds timestamped *session* folders; each session is split into raw
grayscale AVI chunks `video_0.avi, video_1.avi, …` (5 s each, 500 fps, ~1.2 GB per chunk),
with matching `timestamps_N.dat`. Frame resolution can differ between sessions. OpenCV can't
read these raw AVIs, so everything goes through **ffmpeg**.

```
<ANIMAL_DIR>/2025-08-15_12-10-58/video_0.avi
                                 video_1.avi
                                 ...
```

Set `ev.ANIMAL_DIR` at the top of each notebook (default `/mnt/at-storageB1_I/EyeVideo/AT-B1NO1`).

## Contents

- **`eyevideo.py`** — all shared helpers: session/frame IO, clip extraction, pupil detection
  (robust ellipse + the online dark-centroid tracker), the eye-anchored coordinate frame, and
  the analysis/plots.
- **`single_session_inspection.ipynb`** — one session: play a clip, static frames across the
  session (drift check), pupil-tracking example frames, and the pupil cloud in eye coordinates.
- **`across_sessions_eyeframe.ipynb`** — click eye landmarks per day, QC tracking, and compare
  pupil position (eye frame) between `centered` and `biased` day-groups.

## Method (why the eye frame)

Pupil position *in the image* mixes eye rotation (real gaze) with translation from the per-day
crop/zoom and small head-fixation differences, so it isn't comparable across days. We build an
eye-anchored frame from a few landmarks clicked around the eye opening — `u` runs corner-to-corner
(−1 … +1, 0 = centered), `v` is vertical — which is invariant to crop/translation/zoom. Expressing
the pupil in this frame:

- bias present between conditions in eye coordinates → **real gaze bias**;
- bias in image position but not in eye coordinates → **crop / head / processing artifact**.

The online tracker (`get_pupil_online`: centroid of dark pixels in a threshold band) is run beside
a shape-based ellipse detector on the same frames; agreement means the online tracker is faithful.

## Requirements

- `ffmpeg` / `ffprobe` on `PATH`
- Python: `numpy`, `opencv-python`, `matplotlib`, and Jupyter (`ipython`)
- For clicking landmarks: an interactive matplotlib backend (`%matplotlib widget` or `qt`)

## Landmarks

Clicked landmarks are cached in `eye_landmarks.json` (one entry per date), so you only click each
day once; both notebooks share the file.
