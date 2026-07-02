"""B2 booth report: 7 recent sessions, Part A (tracking quality + pupil-size threshold)
and Part B offline-vs-online agreement only. No landmarks / eye-frame / conditions.
Writes results_b2.json and figures_b2/*.png."""
import os, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import eyevideo as ev

ev.ANIMAL_DIR = "/mnt/at-storageB2_I/EyeVideo/AT-B2NO1"
N = 1000
HIGH = 50.0
dates = ['2026-03-16', '2026-04-06', '2026-04-27', '2026-05-13', '2026-05-27', '2026-06-11', '2026-06-25']
EX = [dates[0], dates[2], dates[4], dates[6]]
FIG = "figures_b2"; os.makedirs(FIG, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log(f"tracking {len(dates)} B2 sessions at N={N} ...")
ev.track_dates(dates, n=N, high=HIGH)
log("tracking done")

# --- Part A: tracking quality ---
ev.show_tracking_examples(EX, k=5)
plt.savefig(f"{FIG}/tracking_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")

# --- Part A: pupil size and threshold (single group) ---
ev.pupil_size_histogram(conditions={"AT-B2NO1": dates}, n=N, high=HIGH)
plt.savefig(f"{FIG}/pupil_size_histogram.png", dpi=110, bbox_inches="tight"); plt.close("all")
ev.show_pupil_sizes(dates, n=N, high=HIGH)
plt.savefig(f"{FIG}/pupil_sizes.png", dpi=110, bbox_inches="tight"); plt.close("all")
ev.discrepancy_vs_pupilsize(dates, n=N, high=HIGH)
plt.savefig(f"{FIG}/discrepancy_vs_pupilsize.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("Part A figures done")

# --- Part B: offline vs online agreement (open frames) ---
ev.tracker_agreement(dates, n=N, high=HIGH)
plt.savefig(f"{FIG}/tracker_agreement.png", dpi=110, bbox_inches="tight"); plt.close("all")
ev.show_below_diagonal_examples(dates, k=6)
plt.savefig(f"{FIG}/discrepancy_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")
ev.error_degrees_violin(dates, n=N, high=HIGH)
plt.savefig(f"{FIG}/error_degrees_violin.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("Part B figures done")

# summary numbers on open frames
ex, ox, ey, oy, nd = [], [], [], [], []
for d in dates:
    r = ev.track_both(ev.session_for_date(d), N, HIGH); sel = ev.open_frames(r)
    ex += list(r["ell"][sel, 0]); ox += list(r["onl"][sel, 0])
    ey += list(r["ell"][sel, 1]); oy += list(r["onl"][sel, 1]); nd += list(r["ndark"])
ex, ox, ey, oy, nd = map(np.array, (ex, ox, ey, oy, nd))
closed = {d: ev.closed_fraction([d], N, HIGH)[d]["closed_fraction"] for d in dates}
out = {"N": N, "high": HIGH, "open_min": ev.OPEN_MIN, "animal": "AT-B2NO1", "dates": dates,
       "n_open_total": int(len(ex)),
       "agreement": {"corr_x": float(np.corrcoef(ex, ox)[0, 1]), "corr_y": float(np.corrcoef(ey, oy)[0, 1]),
                     "median_abs_dx_px": float(np.median(np.abs(ex-ox))), "median_abs_dy_px": float(np.median(np.abs(ey-oy)))},
       "closed_fraction": closed,
       "ndark_median": float(np.median(nd))}
json.dump(out, open("results_b2.json", "w"), indent=2)
log("wrote results_b2.json and figures_b2/. DONE")
