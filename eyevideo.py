"""
Eye-video analysis helpers (shared by the single-session and across-session notebooks).

Data layout: an animal folder (e.g. /mnt/at-storageB1_I/EyeVideo/AT-B1NO1) holds timestamped
session folders; each session is split into raw grayscale AVI chunks video_0.avi ... (5 s each,
500 fps). Resolution varies between sessions. OpenCV can't read these AVIs, so we use ffmpeg.

Set ANIMAL_DIR (module global) before use, or pass animal_dir=/full paths.
"""
import os, glob, subprocess, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ANIMAL_DIR = "/mnt/at-storageB1_I/EyeVideo/AT-B1NO1"
CHUNK_SECONDS = 5.0
LANDMARK_FILE = "eye_landmarks.json"

_SIZE_CACHE = {}
_TRK = {}

# ---------------------------------------------------------------- sessions & IO
def chunk_index(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])

def session_chunks(session_dir):
    files = glob.glob(os.path.join(session_dir, "video_*.avi"))
    return sorted(((chunk_index(f), f) for f in files))

def session_duration(session_dir):
    n = len(session_chunks(session_dir))
    return n * CHUNK_SECONDS, n

def list_sessions(animal_dir=None):
    a = animal_dir or ANIMAL_DIR
    return [p for p in sorted(glob.glob(os.path.join(a, "*")))
            if os.path.isdir(p) and glob.glob(os.path.join(p, "video_*.avi"))]

def pick_session(name=None, animal_dir=None):
    ss = list_sessions(animal_dir)
    if not ss:
        raise FileNotFoundError("no sessions with video")
    if name is None:
        import random
        sel = random.choice(ss); print("random session:", os.path.basename(sel)); return sel
    m = [s for s in ss if name in os.path.basename(s)]
    if not m:
        raise ValueError(f"no session matching {name!r}")
    if len(m) > 1:
        print(f"{len(m)} matches for {name!r}, using {os.path.basename(m[0])}")
    return m[0]

def session_for_date(date, animal_dir=None):
    """Resolve 'YYYY-MM-DD' to its session folder (longest, if several that day)."""
    ss = [s for s in list_sessions(animal_dir) if os.path.basename(s).startswith(str(date))]
    if not ss:
        raise ValueError(f"no session on {date}")
    if len(ss) > 1:
        ss = sorted(ss, key=lambda s: session_duration(s)[1], reverse=True)
    return ss[0]

def frame_size(session_dir):
    if session_dir not in _SIZE_CACHE:
        f = session_chunks(session_dir)[0][1]
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", f],
                             capture_output=True, text=True).stdout.strip()
        _SIZE_CACHE[session_dir] = tuple(int(v) for v in out.split("x"))
    return _SIZE_CACHE[session_dir]

def grab_frame(session_dir, t):
    """One grayscale frame (H, W) at time t seconds; None past the end."""
    chunks = dict(session_chunks(session_dir))
    ci = int(min(t // CHUNK_SECONDS, len(chunks) - 1))
    offset = t - ci * CHUNK_SECONDS
    w, h = frame_size(session_dir)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{offset:.4f}", "-i", chunks[ci],
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    if len(raw) < w * h:
        return None
    return np.frombuffer(raw[:w * h], np.uint8).reshape(h, w)

# ---------------------------------------------------------------- video clips
def extract_clip(session_dir, duration=20.0, start=None, slowdown=1.0,
                 display_fps=30, crf=23, out_dir="clips"):
    """Extract `duration` s (random start if None) to a small mp4; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    chunks = session_chunks(session_dir); total = len(chunks) * CHUNK_SECONDS
    duration = min(duration, total)
    if start is None:
        import random
        start = random.uniform(0, max(0.0, total - duration))
    end = min(start + duration, total)
    first = int(start // CHUNK_SECONDS); last = int((end - 1e-6) // CHUNK_SECONDS)
    used = [p for idx, p in chunks if first <= idx <= last]
    offset = start - first * CHUNK_SECONDS
    import tempfile
    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for p in used:
        lst.write(f"file '{os.path.abspath(p)}'\n")
    lst.close()
    out = os.path.join(out_dir, f"{os.path.basename(session_dir)}_t{start:.1f}s_d{duration:.0f}s_x{slowdown:g}.mp4")
    vf = f"setpts=PTS*{slowdown},fps={display_fps}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst.name,
                    "-ss", f"{offset:.4f}", "-t", f"{duration:.4f}", "-vf", vf, "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out], check=True)
    os.unlink(lst.name)
    print(f"{os.path.basename(session_dir)} [{start:.1f},{end:.1f}]s x{slowdown:g} -> {out} "
          f"({os.path.getsize(out)/1e6:.1f} MB)")
    return out

# ---------------------------------------------------------------- static frame grids
def drift_grid(session_dir, rows=5, cols=5, grid_step=100, figscale=2.6):
    """rows*cols frames equally spaced across the whole session, with a fixed reference grid."""
    total, _ = session_duration(session_dir); n = rows * cols
    times = np.linspace(0, total, n, endpoint=False)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figscale, rows*figscale), squeeze=False)
    for ax, t in zip(axes.ravel(), times):
        g = grab_frame(session_dir, t)
        if g is not None:
            h, w = g.shape
            ax.imshow(g, cmap="gray", vmin=0, vmax=255)
            for x in range(0, w, grid_step): ax.axvline(x, color="lime", lw=0.5, alpha=0.55)
            for y in range(0, h, grid_step): ax.axhline(y, color="lime", lw=0.5, alpha=0.55)
            ax.axvline(w/2, color="red", lw=0.7); ax.axhline(h/2, color="red", lw=0.7)
            ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_title(f"{t/60:.1f} min", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(os.path.basename(session_dir) + "  drift check")
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

# ---------------------------------------------------------------- pupil detection
def detect_pupil(g, min_circ=0.45):
    """Darkest roughly-circular blob -> dict(cx,cy,r,circ), or None if eye looks closed."""
    e = detect_pupil_ellipse(g, min_circ)
    if e is None:
        return None
    (cx, cy), (MA, ma), ang = e
    return dict(cx=cx, cy=cy, r=(MA+ma)/4, circ=None)

def detect_pupil_ellipse(g, min_circ=0.45):
    """Return the fitted pupil ellipse ((cx,cy),(MA,ma),angle) or None."""
    H, W = g.shape; b = cv2.GaussianBlur(g, (5, 5), 0)
    thr = min(max(np.percentile(b, 3), 15), 80)
    m = (b <= thr).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, bs = None, -1
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.002*H*W or a > 0.25*H*W: continue
        per = cv2.arcLength(c, True)
        if per == 0: continue
        circ = 4*np.pi*a/(per*per)
        if circ < min_circ: continue
        if circ > bs: bs, best = circ, c
    if best is None or len(best) < 5:
        return None
    return cv2.fitEllipse(best)

def get_pupil_online(image, low_threshold=0.0, high_threshold=50.0):
    """Online tracker: centroid of all dark pixels in (low, high], after a 3x3 erode."""
    low = cv2.threshold(image, low_threshold, 255, cv2.THRESH_BINARY)[1]
    high = cv2.threshold(image, high_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    m = cv2.erode(cv2.bitwise_and(low, high), np.ones((3, 3), np.uint8), 1)
    mo = cv2.moments(m)
    if mo["m00"] == 0:
        return None, None
    return mo["m10"]/mo["m00"], mo["m01"]/mo["m00"]

def track_both(session_dir, n=120, high=50.0):
    """Robust ellipse pupil AND online centroid on the same n frames (cached)."""
    key = (session_dir, n, high)
    if key in _TRK:
        return _TRK[key]
    total, _ = session_duration(session_dir); acc = None; PE = []; PO = []
    for t in np.linspace(0, total, n, endpoint=False):
        g = grab_frame(session_dir, t)
        if g is None: continue
        d = detect_pupil(g)
        if d is None: continue
        ox, oy = get_pupil_online(g, 0.0, high)
        if ox is None: continue
        if acc is None: acc = np.zeros(g.shape, float)
        if g.shape != acc.shape: continue
        acc += g; PE.append((d["cx"], d["cy"])); PO.append((ox, oy))
    W, H = frame_size(session_dir)
    r = dict(ell=np.array(PE), onl=np.array(PO), n=len(PE), W=W, H=H,
             mean=(acc/len(PE)).astype(np.uint8) if PE else None)
    _TRK[key] = r
    return r

# ---------------------------------------------------------------- eye-anchored frame
def eye_frame(pts):
    """Eye frame from clicked eye-opening points (>=3): u along fissure, v vertical."""
    P = np.array(pts, float); c = P.mean(0)
    val, vec = np.linalg.eigh(np.cov((P - c).T))
    ud = vec[:, val.argmax()]; ud = -ud if ud[0] < 0 else ud
    vd = np.array([-ud[1], ud[0]]); vd = -vd if vd[1] < 0 else vd
    pu = (P - c) @ ud; pv = (P - c) @ vd
    return dict(c=c, ud=ud, vd=vd, u0=(pu.max()+pu.min())/2, hu=(pu.max()-pu.min())/2,
                v0=(pv.max()+pv.min())/2, hv=(pv.max()-pv.min())/2)

def to_eye(P, F):
    """Image point -> eye-frame (u, v): -1..+1 across the clicked extent, 0 = center."""
    d = np.array(P, float) - F["c"]
    return ((d @ F["ud"]) - F["u0"])/F["hu"], ((d @ F["vd"]) - F["v0"])/F["hv"]

# ---------------------------------------------------------------- landmarks
def load_landmarks():
    if os.path.exists(LANDMARK_FILE):
        return {k: [tuple(p) for p in v] for k, v in json.load(open(LANDMARK_FILE)).items()}
    return {}

def save_landmarks(LM):
    json.dump({k: [list(p) for p in v] for k, v in LM.items()}, open(LANDMARK_FILE, "w"), indent=1)

LANDMARKS = load_landmarks()

def show_mean_grid(date, n=120):
    """Mean eye image with a pixel grid (to read off / plan landmark coordinates)."""
    r = track_both(session_for_date(date), n); m = r["mean"]; H, W = m.shape
    fig, ax = plt.subplots(figsize=(10, 7)); ax.imshow(m, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks(np.arange(0, W, 50)); ax.set_yticks(np.arange(0, H, 50))
    ax.grid(True, color="lime", lw=0.4, alpha=0.6); ax.set_title(f"{date}  {W}x{H}"); plt.show()

def click_landmarks(date, n_points=8, n=120):
    """Click n_points around the eye opening on the mean image; saved to eye_landmarks.json.
    Needs an interactive backend: run  %matplotlib widget  (or qt) in a cell first."""
    r = track_both(session_for_date(date), n)
    if r["mean"] is None:
        raise ValueError("no pupil frames for " + date)
    fig, ax = plt.subplots(figsize=(10, 7)); ax.imshow(r["mean"], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{date}: click {n_points} points around the eye opening, then press Enter")
    pts = plt.ginput(n_points, timeout=0); plt.close(fig)
    LANDMARKS[date] = [(float(x), float(y)) for x, y in pts]; save_landmarks(LANDMARKS)
    print(f"saved {len(pts)} landmarks for {date}"); return LANDMARKS[date]

def show_landmarks(date, n=120):
    """Overlay saved landmarks + eye-frame axes on the mean image to verify them."""
    r = track_both(session_for_date(date), n); F = eye_frame(LANDMARKS[date])
    fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(r["mean"], cmap="gray", vmin=0, vmax=255)
    P = np.array(LANDMARKS[date]); ax.plot(np.append(P[:,0],P[0,0]), np.append(P[:,1],P[0,1]), "y-o", ms=4)
    ax.arrow(*F["c"], *(F["ud"]*F["hu"]), color="red", width=2)
    ax.arrow(*F["c"], *(F["vd"]*F["hv"]), color="lime", width=2)
    ax.set_title(f"{date} eye frame (red=u, green=v)"); ax.set_xticks([]); ax.set_yticks([]); plt.show()

# ---------------------------------------------------------------- QC & analysis plots
def show_tracking_examples(dates, k=5, high=50.0, search=40):
    """Rows = dates, cols = k example open-eye frames with both trackers overlaid."""
    if isinstance(dates, str): dates = [dates]
    fig, axes = plt.subplots(len(dates), k, figsize=(k*2.8, len(dates)*2.4), squeeze=False)
    for r, d in enumerate(dates):
        sdir = session_for_date(d); total, _ = session_duration(sdir); found = []
        for t in np.linspace(0, total, search, endpoint=False):
            g = grab_frame(sdir, t)
            if g is None: continue
            e = detect_pupil_ellipse(g)
            if e is None: continue
            found.append((t, g, e))
            if len(found) >= k: break
        for c in range(k):
            ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
            if c >= len(found): ax.axis("off"); continue
            t, g, e = found[c]; (cx, cy), (MA, ma), ang = e
            ax.imshow(g, cmap="gray", vmin=0, vmax=255)
            ax.add_patch(Ellipse((cx, cy), MA, ma, angle=ang, fill=False, ec="cyan", lw=1.6))
            ax.plot(cx, cy, "+", c="cyan", ms=9, mew=1.6)
            ox, oy = get_pupil_online(g, 0.0, high)
            if ox is not None: ax.plot(ox, oy, "x", c="orange", ms=9, mew=1.8)
            if c == 0: ax.set_ylabel(d, fontsize=9)
            ax.set_title(f"{t/60:.0f}m", fontsize=7)
    plt.tight_layout(); plt.show()

def pupil_cloud_eyeframe(date, n=120, high=50.0):
    """Single session: mean eye with landmarks/axes + pupil clouds (both trackers) in image space,
    and the pupil-center scatter in eye-frame (u, v)."""
    if date not in LANDMARKS:
        raise ValueError(f"no landmarks for {date} - click_landmarks('{date}') first")
    r = track_both(session_for_date(date), n, high); F = eye_frame(LANDMARKS[date])
    ue = np.array([to_eye(p, F) for p in r["ell"]]); uo = np.array([to_eye(p, F) for p in r["onl"]])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].imshow(r["mean"], cmap="gray", vmin=0, vmax=255)
    P = np.array(LANDMARKS[date]); ax[0].plot(np.append(P[:,0],P[0,0]), np.append(P[:,1],P[0,1]), "y-", lw=1)
    ax[0].scatter(r["ell"][:,0], r["ell"][:,1], s=8, c="cyan", alpha=0.5, label="robust")
    ax[0].scatter(r["onl"][:,0], r["onl"][:,1], s=8, c="orange", alpha=0.5, label="online")
    ax[0].arrow(*F["c"], *(F["ud"]*F["hu"]), color="red", width=2)
    ax[0].arrow(*F["c"], *(F["vd"]*F["hv"]), color="lime", width=2)
    ax[0].set_title(f"{date}  (image space)"); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].legend(fontsize=8)
    ax[1].scatter(ue[:,0], ue[:,1], s=12, c="cyan", alpha=0.6, label="robust")
    ax[1].scatter(uo[:,0], uo[:,1], s=12, c="orange", alpha=0.6, label="online")
    ax[1].plot(np.median(ue[:,0]), np.median(ue[:,1]), "*", c="blue", ms=16, mec="w")
    ax[1].axvline(0, color="k", ls=":"); ax[1].axhline(0, color="k", ls=":")
    ax[1].set_xlim(-1.2, 1.2); ax[1].set_ylim(1.2, -1.2); ax[1].set_aspect("equal")
    ax[1].set_xlabel("u (-1..+1 corner, 0=center)"); ax[1].set_ylabel("v (top..bottom)")
    ax[1].set_title("pupil in eye frame"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()
    print(f"{date}: median eye-frame u robust={np.median(ue[:,0]):+.3f} online={np.median(uo[:,0]):+.3f}  n={r['n']}")
    return dict(ue=ue, uo=uo)

def compare_conditions(centered, biased, n=120, high=50.0, landmarks=None):
    """Compare pupil u (eye frame) between two groups of dates, for both trackers."""
    LM = landmarks if landmarks is not None else LANDMARKS
    conds = [("centered", centered, "tab:green"), ("biased", biased, "tab:red")]
    pooled = {c[0]: {"ell": [], "onl": []} for c in conds}; perday = []
    for name, dates, _ in conds:
        for d in dates:
            if d not in LM:
                raise ValueError(f"no landmarks for {d} - click_landmarks('{d}') first")
            r = track_both(session_for_date(d), n, high); F = eye_frame(LM[d])
            ue = [to_eye(p, F)[0] for p in r["ell"]]; uo = [to_eye(p, F)[0] for p in r["onl"]]
            pooled[name]["ell"] += ue; pooled[name]["onl"] += uo
            perday.append((name, d, float(np.median(ue)), float(np.median(uo)), r["n"]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, _, col in conds:
        ax[0].hist(pooled[name]["ell"], bins=25, range=(-1, 1), alpha=0.5, color=col, label=name)
        ax[1].hist(pooled[name]["onl"], bins=25, range=(-1, 1), alpha=0.5, color=col, label=name)
    for a, ttl in zip(ax, ["robust ellipse pupil", "online centroid"]):
        a.axvline(0, color="k", ls=":"); a.set_title(ttl); a.legend(fontsize=8)
        a.set_xlabel("pupil u in eye frame (-1 .. +1 corner, 0 = centered)")
    plt.tight_layout(); plt.show()
    print(f"{'cond':<10}{'date':<12}{'u_robust':>9}{'u_online':>9}{'n':>6}")
    for row in perday:
        print(f"{row[0]:<10}{row[1]:<12}{row[2]:>9.3f}{row[3]:>9.3f}{row[4]:>6d}")
    for name, _, _ in conds:
        e = np.array(pooled[name]["ell"]); o = np.array(pooled[name]["onl"])
        print(f"POOLED {name:<9} robust u median={np.median(e):+.3f}  online u median={np.median(o):+.3f}  N={len(e)}")
    return pooled, perday
