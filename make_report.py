"""Run the across-session eye-frame analysis at N=1000 and dump results + figures.
All eye-frame comparisons use frames with pupil size (ndark) above ev.OPEN_MIN.
Writes results.json and figures/*.png; REPORT.md is assembled separately."""
import os, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import eyevideo as ev

ev.ANIMAL_DIR = "/mnt/at-storageB1_I/EyeVideo/AT-B1NO1"
N = 1000
HIGH = 50.0            # booth 1
ev.OPEN_MIN = 5000     # booth-1 openness threshold (from the pupil-size histogram)
centered = ['2026-05-27', '2026-04-30', '2026-06-11', '2026-06-09', '2026-03-02', '2026-02-04', '2026-02-13']
biased   = ['2026-06-04', '2026-05-20', '2026-05-12', '2026-04-17', '2026-03-19', '2026-04-13', '2026-04-07']
EX = [centered[0], centered[3], biased[0], biased[3]]
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# 1) track once at N (cached)
log(f"tracking {len(centered)+len(biased)} sessions at N={N} ...")
ev.track_dates(centered + biased, n=N, high=HIGH)
log("tracking done")

# --- A. qualitative tracking inspection ---
ev.show_tracking_examples(EX, k=5)
plt.savefig(f"{FIG}/tracking_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")
ev.show_eyeframes(EX, n=N, cols=2)
plt.savefig(f"{FIG}/eyeframe_coordinates.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("tracking-quality figures done")

# --- B. pupil size and threshold ---
ev.pupil_size_histogram(centered, biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/pupil_size_histogram.png", dpi=110, bbox_inches="tight"); plt.close("all")
ev.show_pupil_sizes(centered + biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/pupil_sizes.png", dpi=110, bbox_inches="tight"); plt.close("all")
ev.discrepancy_vs_pupilsize(centered + biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/discrepancy_vs_pupilsize.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("pupil-size figures done")

# --- C. real analysis on open frames (ndark > OPEN_MIN) ---
# offline vs online agreement (open frames)
ev.tracker_agreement(centered + biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/tracker_agreement.png", dpi=110, bbox_inches="tight"); plt.close("all")
# example frames of moderate below-diagonal discrepancy
ev.show_below_diagonal_examples(centered + biased, k=6)
plt.savefig(f"{FIG}/discrepancy_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")
# distribution of on-screen error in degrees
ev.error_degrees_violin(centered + biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/error_degrees_violin.png", dpi=110, bbox_inches="tight"); plt.close("all")
ex, ox2, ey, oy2 = [], [], [], []
for d in centered + biased:
    r = ev.track_both(ev.session_for_date(d), N, HIGH); sel = ev.open_frames(r)
    ex += list(r["ell"][sel, 0]); ox2 += list(r["onl"][sel, 0])
    ey += list(r["ell"][sel, 1]); oy2 += list(r["onl"][sel, 1])
ex, ox2, ey, oy2 = map(np.array, (ex, ox2, ey, oy2))
agree = {"corr_x": float(np.corrcoef(ex, ox2)[0, 1]), "corr_y": float(np.corrcoef(ey, oy2)[0, 1]),
         "median_abs_dx_px": float(np.median(np.abs(ex - ox2))),
         "median_abs_dy_px": float(np.median(np.abs(ey - oy2)))}
log("agreement (open frames) done")

# --- eye-frame comparison on open frames (ndark > OPEN_MIN) ---
res = ev.compare_conditions(centered, biased, n=N, high=HIGH, bins=60, summary="mean", open_only=True)
plt.savefig(f"{FIG}/eyeframe_histograms.png", dpi=110, bbox_inches="tight"); plt.close("all")
daysum = res["daysum"]; perday = res["perday"]
welch = {}
for tr in ["ell", "onl"]:
    for ax in ["u", "v"]:
        _, st, p = ev._group_test([daysum["centered"][tr][ax], daysum["biased"][tr][ax]])
        welch[f"{tr}/{ax}"] = {"t": st, "p": p,
                               "centered_mean": float(np.mean(daysum["centered"][tr][ax])),
                               "biased_mean": float(np.mean(daysum["biased"][tr][ax]))}
# robustness: same test on ALL detected frames (no openness filter)
res_all = ev.compare_conditions(centered, biased, n=N, high=HIGH, open_only=False); plt.close("all")
welch_allframes = {}
for tr in ["ell", "onl"]:
    for ax in ["u", "v"]:
        _, st, p = ev._group_test([res_all["daysum"]["centered"][tr][ax], res_all["daysum"]["biased"][tr][ax]])
        welch_allframes[f"{tr}/{ax}"] = {"t": st, "p": p}
log("compare_conditions done")

# tracker discrepancy by condition (open frames)
ev.compare_agreement(centered, biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/tracker_discrepancy.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("compare_agreement done")

# image-frame vs eye-frame contrast (open frames)
img_norm = {"centered": [], "biased": []}; eye_u = {"centered": [], "biased": []}
for name, dates in [("centered", centered), ("biased", biased)]:
    for d in dates:
        r = ev.track_both(ev.session_for_date(d), N, HIGH); F = ev.eye_frame(ev.LANDMARKS[d])
        sel = ev.open_frames(r)
        img_norm[name].append(float(np.mean(r["ell"][sel, 0] / r["W"])))
        eye_u[name].append(float(np.mean([ev.to_eye(p, F)[0] for p in r["ell"][sel]])))
def diff(dic): return float(np.mean(dic["biased"]) - np.mean(dic["centered"]))
_, _, p_img = ev._group_test([img_norm["biased"], img_norm["centered"]])
_, _, p_eye = ev._group_test([eye_u["biased"], eye_u["centered"]])
contrast = {
    "image_norm_x": {"centered_mean": float(np.mean(img_norm["centered"])),
                     "biased_mean": float(np.mean(img_norm["biased"])),
                     "diff_biased_minus_centered": diff(img_norm), "welch_p": p_img},
    "eyeframe_u": {"centered_mean": float(np.mean(eye_u["centered"])),
                   "biased_mean": float(np.mean(eye_u["biased"])),
                   "diff_biased_minus_centered": diff(eye_u), "welch_p": p_eye},
}

out = {
    "N": N, "high": HIGH, "open_min": ev.OPEN_MIN,
    "centered": centered, "biased": biased,
    "per_session": [{"cond": r[0], "date": r[1], "u_rob": r[2], "v_rob": r[3],
                     "u_onl": r[4], "v_onl": r[5], "n": r[6]} for r in perday],
    "welch_session_level": welch,
    "welch_allframes": welch_allframes,
    "tracker_agreement": agree,
    "image_vs_eyeframe": contrast,
}
json.dump(out, open("results.json", "w"), indent=2)
log("wrote results.json and figures/. DONE")
