"""Report 2 — booth-1 gaze-bias investigation (eye-anchored frame).
Focuses only on the gaze-bias question (real shift vs online-tracker artifact);
tracking quality / pupil-size / accuracy are Report 1 (make_report_accuracy.py).
Writes results.json and figures_gaze_bias/{eyeframe_coordinates,eyeframe_histograms,tracker_discrepancy}.png."""
import os, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import eyevideo as ev

ev.ANIMAL_DIR = "/mnt/at-storageB1_I/EyeVideo/AT-B1NO1"
N = 1000
HIGH = 50.0            # booth 1
ev.OPEN_MIN = 5000     # booth-1 openness threshold
centered = ['2026-05-27', '2026-04-30', '2026-06-11', '2026-06-09', '2026-03-02', '2026-02-04', '2026-02-13']
biased   = ['2026-06-04', '2026-05-20', '2026-05-12', '2026-04-17', '2026-03-19', '2026-04-13', '2026-04-07']
EX = [centered[0], centered[3], biased[0], biased[3]]
FIG = "figures_gaze_bias"; os.makedirs(FIG, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

ev.track_dates(centered + biased, n=N, high=HIGH)
log("tracking done")

# eye-anchored coordinate frame (setup)
ev.show_eyeframes(EX, n=N, cols=2)
plt.savefig(f"{FIG}/eyeframe_coordinates.png", dpi=110, bbox_inches="tight"); plt.close("all")

# main result: eye-frame position, centered vs biased (open frames)
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
res_all = ev.compare_conditions(centered, biased, n=N, high=HIGH, open_only=False); plt.close("all")
welch_allframes = {f"{tr}/{ax}": {"p": ev._group_test([res_all["daysum"]["centered"][tr][ax], res_all["daysum"]["biased"][tr][ax]])[2]}
                   for tr in ["ell", "onl"] for ax in ["u", "v"]}
log("compare_conditions done")

# does the online-offline discrepancy differ between conditions? (artifact check)
ev.compare_agreement(centered, biased, n=N, high=HIGH)
plt.savefig(f"{FIG}/tracker_discrepancy.png", dpi=110, bbox_inches="tight"); plt.close("all")
log("compare_agreement done")

# eye-frame u vs raw image-x, between conditions (open frames)
img_norm = {"centered": [], "biased": []}; eye_u = {"centered": [], "biased": []}
ex, ox = [], []
for name, dates in [("centered", centered), ("biased", biased)]:
    for d in dates:
        r = ev.track_both(ev.session_for_date(d), N, HIGH); F = ev.eye_frame(ev.LANDMARKS[d]); sel = ev.open_frames(r)
        img_norm[name].append(float(np.mean(r["ell"][sel, 0] / r["W"])))
        eye_u[name].append(float(np.mean([ev.to_eye(p, F)[0] for p in r["ell"][sel]])))
        ex += list(r["ell"][sel, 0]); ox += list(r["onl"][sel, 0])
_, _, p_img = ev._group_test([img_norm["biased"], img_norm["centered"]])
_, _, p_eye = ev._group_test([eye_u["biased"], eye_u["centered"]])
contrast = {"image_norm_x": {"diff_biased_minus_centered": float(np.mean(img_norm["biased"])-np.mean(img_norm["centered"])), "welch_p": p_img},
            "eyeframe_u": {"diff_biased_minus_centered": float(np.mean(eye_u["biased"])-np.mean(eye_u["centered"])), "welch_p": p_eye}}
agree_corr_x = float(np.corrcoef(ex, ox)[0, 1])   # brief, for the conclusion (full accuracy is Report 1)

out = {"N": N, "high": HIGH, "open_min": ev.OPEN_MIN, "centered": centered, "biased": biased,
       "per_session": [{"cond": r[0], "date": r[1], "u_rob": r[2], "v_rob": r[3], "u_onl": r[4], "v_onl": r[5], "n": r[6]} for r in perday],
       "welch_session_level": welch, "welch_allframes": welch_allframes,
       "image_vs_eyeframe": contrast, "online_offline_corr_x": agree_corr_x}
json.dump(out, open("results.json", "w"), indent=2)
log("wrote results.json and figures_gaze_bias/. DONE")
