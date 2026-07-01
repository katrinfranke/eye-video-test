"""Run the across-session eye-frame analysis at N=500 and dump results + figures.
Writes results.json and figures/*.png; REPORT.md is assembled separately."""
import os, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import eyevideo as ev

ev.ANIMAL_DIR = "/mnt/at-storageB1_I/EyeVideo/AT-B1NO1"
N = 1000
HIGH = 50.0
centered = ['2026-05-27', '2026-04-30', '2026-06-11', '2026-06-09', '2026-03-02', '2026-02-04', '2026-02-13']
biased   = ['2026-06-04', '2026-05-20', '2026-05-12', '2026-04-17', '2026-03-19', '2026-04-13', '2026-04-07']
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
t0 = time.time()

def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# 1) track once at N=500 (slow part)
log(f"tracking {len(centered)+len(biased)} sessions at N={N} ...")
ev.track_dates(centered + biased, n=N, high=HIGH)
log("tracking done")

# 2) eye-frame comparison (main result) + figure
res = ev.compare_conditions(centered, biased, n=N, high=HIGH, bins=60, summary="mean")
plt.savefig(f"{FIG}/eyeframe_histograms.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("compare_conditions done")

daysum = res["daysum"]; perday = res["perday"]

# 3b) example frames with ellipse + cross (2 centered + 2 biased sessions)
EX = [centered[0], centered[3], biased[0], biased[3]]
ev.show_tracking_examples(EX, k=5)
plt.savefig(f"{FIG}/tracking_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("tracking_examples figure done")

# 3c) eye-frame coordinate system on 4 example sessions
ev.show_eyeframes(EX, n=N, cols=2)
plt.savefig(f"{FIG}/eyeframe_coordinates.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("eyeframe_coordinates figure done")

# 4) tracker discrepancy by condition + figure
ev.compare_agreement(centered, biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/tracker_discrepancy.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("compare_agreement done")

# 5) tracker agreement scatter + summary numbers
ev.tracker_agreement(centered + biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/tracker_agreement.png", dpi=110, bbox_inches="tight"); plt.close("all")

# 5b) highest-discrepancy frames with contributing pixels highlighted
ev.show_discrepancy_examples(centered + biased, k=8, n=N, high=HIGH, by="both")
plt.savefig(f"{FIG}/discrepancy_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("discrepancy_examples figure done")

allx_e, allx_o, ally_e, ally_o, dxs, dys = [], [], [], [], [], []
for d in centered + biased:
    r = ev.track_both(ev.session_for_date(d), N, HIGH)
    allx_e += list(r["ell"][:, 0]); allx_o += list(r["onl"][:, 0])
    ally_e += list(r["ell"][:, 1]); ally_o += list(r["onl"][:, 1])
    dxs += list(r["onl"][:, 0] - r["ell"][:, 0]); dys += list(r["onl"][:, 1] - r["ell"][:, 1])
agree = {
    "corr_x": float(np.corrcoef(allx_e, allx_o)[0, 1]),
    "corr_y": float(np.corrcoef(ally_e, ally_o)[0, 1]),
    "median_abs_dx_px": float(np.median(np.abs(dxs))),
    "median_abs_dy_px": float(np.median(np.abs(dys))),
}
log("tracker_agreement done")

# 6) image-frame vs eye-frame contrast (how much the eye frame changes the picture)
#    per-session mean normalized image-x (ell_x / W) vs eye-frame u
img_norm = {"centered": [], "biased": []}
eye_u = {"centered": [], "biased": []}
for name, dates in [("centered", centered), ("biased", biased)]:
    for d in dates:
        r = ev.track_both(ev.session_for_date(d), N, HIGH); F = ev.eye_frame(ev.LANDMARKS[d])
        img_norm[name].append(float(np.mean(r["ell"][:, 0] / r["W"])))
        eye_u[name].append(float(np.mean([ev.to_eye(p, F)[0] for p in r["ell"]])))
def diff(dic): return float(np.mean(dic["biased"]) - np.mean(dic["centered"]))
_, _, p_img = ev._group_test([img_norm["biased"], img_norm["centered"]])
_, _, p_eye = ev._group_test([eye_u["biased"], eye_u["centered"]])
contrast = {
    "image_norm_x": {"centered_mean": float(np.mean(img_norm["centered"])),
                     "biased_mean": float(np.mean(img_norm["biased"])),
                     "diff_biased_minus_centered": diff(img_norm), "welch_p": p_img,
                     "per_session": img_norm},
    "eyeframe_u": {"centered_mean": float(np.mean(eye_u["centered"])),
                   "biased_mean": float(np.mean(eye_u["biased"])),
                   "diff_biased_minus_centered": diff(eye_u), "welch_p": p_eye,
                   "per_session": eye_u},
}

# 6b) eye closure: pupil-size examples + closed fraction by condition
ev.show_pupil_sizes(centered + biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/pupil_sizes.png", dpi=110, bbox_inches="tight"); plt.close("all")
cf = {d: v["closed_fraction"] for d, v in ev.closed_fraction(centered + biased, n=N, high=HIGH).items()}
ev.compare_closure(centered, biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/closure_fraction.png", dpi=110, bbox_inches="tight"); plt.close("all")
_, ct, cp = ev._group_test([[cf[d] for d in centered], [cf[d] for d in biased]])
closure = {"per_session": {d: cf[d] for d in centered + biased},
           "centered_mean": float(np.mean([cf[d] for d in centered])),
           "biased_mean": float(np.mean([cf[d] for d in biased])),
           "welch_t": ct, "welch_p": cp, "min_dark": ev.MIN_DARK, "max_dark": ev.MAX_DARK}
log("closure figures done")

# 6c) open-only eye-frame comparison (drop invalid/closed frames)
res_open = ev.compare_conditions(centered, biased, n=N, high=HIGH, open_only=True)
plt.close("all")
open_welch = {}
for tr in ["ell", "onl"]:
    for ax in ["u", "v"]:
        _, st, p = ev._group_test([res_open["daysum"]["centered"][tr][ax], res_open["daysum"]["biased"][tr][ax]])
        open_welch[f"{tr}/{ax}"] = {"t": st, "p": p}
log("open-only comparison done")

# 7) session-level Welch tests per tracker/axis
welch = {}
for tr in ["ell", "onl"]:
    for ax in ["u", "v"]:
        _, st, p = ev._group_test([daysum["centered"][tr][ax], daysum["biased"][tr][ax]])
        welch[f"{tr}/{ax}"] = {"t": st, "p": p,
                               "centered_mean": float(np.mean(daysum["centered"][tr][ax])),
                               "biased_mean": float(np.mean(daysum["biased"][tr][ax]))}

# assemble
out = {
    "N": N, "high": HIGH,
    "centered": centered, "biased": biased,
    "per_session": [{"cond": r[0], "date": r[1], "u_rob": r[2], "v_rob": r[3],
                     "u_onl": r[4], "v_onl": r[5], "n": r[6]} for r in perday],
    "welch_session_level": welch,
    "welch_session_level_open_only": open_welch,
    "tracker_agreement": agree,
    "image_vs_eyeframe": contrast,
    "closure": closure,
}
json.dump(out, open("results.json", "w"), indent=2)
log("wrote results.json and figures/. DONE")
