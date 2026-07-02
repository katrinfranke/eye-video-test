"""Report 1 — reliability of online vs offline eye tracking, across both animals/booths.
Booth-specific thresholds (low=0 always): high=60 & open>6000 (booth 1, washed out);
high=40 & open>11000 (booth 2, in range). Part A (tracking quality + pupil-size) and
Part B (offline-vs-online agreement) for each booth. No eye-frame / conditions.
Writes results_accuracy.json and figures_acc/*.png."""
import os, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import eyevideo as ev

N = 1000
FIG = "figures_acc"; os.makedirs(FIG, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

BOOTHS = {
    "b1": dict(animal="AT-B1NO1", dir="/mnt/at-storageB1_I/EyeVideo/AT-B1NO1", high=50.0, open_min=5000,
               dates=['2026-05-27', '2026-04-30', '2026-06-11', '2026-06-09', '2026-03-02', '2026-02-04', '2026-02-13'],
               ladder=[2000, 3500, 4500, 5000, 6000, 8000, 11000, 14000]),
    "b2": dict(animal="AT-B2NO1", dir="/mnt/at-storageB2_I/EyeVideo/AT-B2NO1", high=40.0, open_min=11000,
               dates=['2026-03-16', '2026-04-06', '2026-04-27', '2026-05-13', '2026-05-27', '2026-06-11', '2026-06-25'],
               ladder=[5000, 8000, 10000, 11000, 12000, 14000, 17000, 20000]),
}

out = {"N": N, "booths": {}}
for key, cfg in BOOTHS.items():
    ev.ANIMAL_DIR = cfg["dir"]; ev.OPEN_MIN = cfg["open_min"]; hi = cfg["high"]; dates = cfg["dates"]
    EX = [dates[0], dates[2], dates[4], dates[6]]
    log(f"{key} ({cfg['animal']}) high={hi} open>{cfg['open_min']}")
    ev.track_dates(dates, n=N, high=hi)
    # Part A
    ev.show_tracking_examples(EX, k=5, high=hi)
    plt.savefig(f"{FIG}/{key}_tracking_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")
    ev.pupil_size_histogram(conditions={cfg["animal"]: dates}, n=N, high=hi)
    plt.savefig(f"{FIG}/{key}_pupil_size_histogram.png", dpi=110, bbox_inches="tight"); plt.close("all")
    ev.show_pupil_sizes(dates, targets=cfg["ladder"], n=N, high=hi)
    plt.savefig(f"{FIG}/{key}_pupil_sizes.png", dpi=110, bbox_inches="tight"); plt.close("all")
    ev.discrepancy_vs_pupilsize(dates, n=N, high=hi)
    plt.savefig(f"{FIG}/{key}_discrepancy_vs_pupilsize.png", dpi=110, bbox_inches="tight"); plt.close("all")
    # Part B
    ev.tracker_agreement(dates, n=N, high=hi)
    plt.savefig(f"{FIG}/{key}_tracker_agreement.png", dpi=110, bbox_inches="tight"); plt.close("all")
    ev.show_below_diagonal_examples(dates, k=6, n=N, high=hi)
    plt.savefig(f"{FIG}/{key}_discrepancy_examples.png", dpi=110, bbox_inches="tight"); plt.close("all")
    ev.error_degrees_violin(dates, n=N, high=hi)
    plt.savefig(f"{FIG}/{key}_error_degrees_violin.png", dpi=110, bbox_inches="tight"); plt.close("all")
    # numbers on open frames
    ex, ox, ey, oy, nd, yrs = [], [], [], [], [], []
    for d in dates:
        r = ev.track_both(ev.session_for_date(d), N, hi); sel = ev.open_frames(r)
        ell = r["ell"][sel]; onl = r["onl"][sel]
        ex += list(ell[:, 0]); ox += list(onl[:, 0]); ey += list(ell[:, 1]); oy += list(onl[:, 1]); nd += list(r["ndark"])
        yrs.append(np.percentile(ell[:, 1], 97.5) - np.percentile(ell[:, 1], 2.5))
    ex, ox, ey, oy, nd = map(np.array, (ex, ox, ey, oy, nd))
    g = 40.0 / np.median(yrs)   # deg per pupil-px (full-height assumption, this booth)
    edx, edy = np.abs(ex-ox)*g, np.abs(ey-oy)*g
    out["booths"][key] = {
        "animal": cfg["animal"], "high": hi, "open_min": cfg["open_min"], "dates": dates,
        "n_open_total": int(len(ex)), "ndark_median": float(np.median(nd)),
        "agreement": {"corr_x": float(np.corrcoef(ex, ox)[0, 1]), "corr_y": float(np.corrcoef(ey, oy)[0, 1]),
                      "median_abs_dx_px": float(np.median(np.abs(ex-ox))), "median_abs_dy_px": float(np.median(np.abs(ey-oy)))},
        "deg_gain": float(g),
        "error_deg": {"median_x": float(np.median(edx)), "median_y": float(np.median(edy)),
                      "frac_gt1_x": float((edx > 1).mean()), "frac_gt1_y": float((edy > 1).mean()),
                      "frac_gt2_x": float((edx > 2).mean()), "frac_gt2_y": float((edy > 2).mean())},
        "closed_fraction": {d: ev.closed_fraction([d], N, hi)[d]["closed_fraction"] for d in dates},
    }
    log(f"{key} done")

json.dump(out, open("results_accuracy.json", "w"), indent=2)
log("wrote results_accuracy.json and figures_acc/. DONE")
