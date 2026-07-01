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

CACHE_DIR = ".track_cache"

def _cache_path(session_dir, n, high):
    import hashlib
    h = hashlib.md5(session_dir.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"{os.path.basename(session_dir)}_{h}_n{n}_h{int(high)}.npz")

def track_both(session_dir, n=120, high=50.0, use_cache=True):
    """Robust ellipse pupil AND online centroid on the same n frames.

    Tracked once per (session, n, high) and stored: in memory and on disk (.track_cache),
    so it survives kernel restarts and is reused by every plot/analysis. Pass use_cache=False
    to force a recompute (e.g. after changing the detector)."""
    key = (session_dir, n, high)
    if use_cache and key in _TRK:
        return _TRK[key]
    cp = _cache_path(session_dir, n, high)
    if use_cache and os.path.exists(cp):
        z = np.load(cp)
        r = dict(ell=z["ell"], onl=z["onl"], t=z["t"], n=int(z["n"]), W=int(z["W"]), H=int(z["H"]),
                 mean=(z["mean"] if z["mean"].size else None))
        _TRK[key] = r
        return r
    total, _ = session_duration(session_dir); acc = None; PE = []; PO = []; T = []
    for t in np.linspace(0, total, n, endpoint=False):
        g = grab_frame(session_dir, t)
        if g is None: continue
        d = detect_pupil(g)
        if d is None: continue
        ox, oy = get_pupil_online(g, 0.0, high)
        if ox is None: continue
        if acc is None: acc = np.zeros(g.shape, float)
        if g.shape != acc.shape: continue
        acc += g; PE.append((d["cx"], d["cy"])); PO.append((ox, oy)); T.append(t)
    W, H = frame_size(session_dir)
    mean = (acc/len(PE)).astype(np.uint8) if PE else None
    r = dict(ell=np.array(PE, float).reshape(-1, 2), onl=np.array(PO, float).reshape(-1, 2),
             t=np.array(T, float), n=len(PE), W=W, H=H, mean=mean)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cp, ell=r["ell"], onl=r["onl"], t=r["t"], n=r["n"], W=r["W"], H=r["H"],
                            mean=(mean if mean is not None else np.array([], np.uint8)))
    _TRK[key] = r
    return r

def track_dates(dates, n=200, high=50.0):
    """Precompute & cache tracking for a list of dates once; returns {date: result}.
    Call this at the top of the notebook so every later plot reuses the stored values."""
    out = {}
    for d in dates:
        r = track_both(session_for_date(d), n, high)
        out[d] = r
        print(f"{d}: {r['n']}/{n} frames with pupil")
    return out

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

class LandmarkClicker:
    """Interactive, non-blocking landmark clicker (works with %matplotlib widget / ipympl).

    Usage in a notebook:
        %matplotlib widget
        clk = ev.LandmarkClicker(DATE)   # click >=5 points around the eye opening
        # ...click on the figure; left-click add, right-click undo last...
        clk.save()                       # writes to eye_landmarks.json
        %matplotlib inline
    """
    def __init__(self, date, n=120):
        r = track_both(session_for_date(date), n)
        if r["mean"] is None:
            raise ValueError("no pupil frames for " + date)
        self.date = date; self.pts = []
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.ax.imshow(r["mean"], cmap="gray", vmin=0, vmax=255)
        self.ax.set_title(f"{date}: left-click points around the eye opening (>=5), "
                          f"right-click to undo, then clk.save()")
        self.marks, = self.ax.plot([], [], "r+-", ms=10, lw=0.8)
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)

    def _onclick(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        if e.button == 3 and self.pts:          # right-click: undo
            self.pts.pop()
        elif e.button == 1:                      # left-click: add
            self.pts.append((float(e.xdata), float(e.ydata)))
        xs = [p[0] for p in self.pts] + ([self.pts[0][0]] if len(self.pts) > 2 else [])
        ys = [p[1] for p in self.pts] + ([self.pts[0][1]] if len(self.pts) > 2 else [])
        self.marks.set_data(xs, ys)
        self.ax.set_title(f"{self.date}: {len(self.pts)} points  (clk.save() when done)")
        self.fig.canvas.draw_idle()

    def save(self):
        if len(self.pts) < 3:
            raise ValueError("click at least 3 points first")
        LANDMARKS[self.date] = list(self.pts); save_landmarks(LANDMARKS)
        print(f"saved {len(self.pts)} landmarks for {self.date}"); return LANDMARKS[self.date]

def set_landmarks(date, points):
    """Manual fallback (no clicking): set landmarks read off show_mean_grid(date)."""
    LANDMARKS[date] = [(float(x), float(y)) for x, y in points]; save_landmarks(LANDMARKS)
    print(f"saved {len(points)} landmarks for {date}"); return LANDMARKS[date]

def has_landmarks(date):
    return date in LANDMARKS and len(LANDMARKS[date]) >= 3

def clicker_if_missing(date, n=120):
    """Return a LandmarkClicker only if `date` has no landmarks yet; else None (skip clicking).
    Use as:  clk = ev.clicker_if_missing(DATE);  then  clk and clk.save()."""
    if has_landmarks(date):
        print(f"landmarks already exist for {date} ({len(LANDMARKS[date])} pts) - skipping clicking")
        return None
    print(f"no landmarks for {date} - click >=5 points, then clk.save()")
    return LandmarkClicker(date, n)

def show_landmarks(date, n=120):
    """Overlay saved landmarks + eye-frame axes on the mean image to verify them."""
    r = track_both(session_for_date(date), n); F = eye_frame(LANDMARKS[date])
    fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(r["mean"], cmap="gray", vmin=0, vmax=255)
    P = np.array(LANDMARKS[date]); ax.plot(np.append(P[:,0],P[0,0]), np.append(P[:,1],P[0,1]), "y-o", ms=4)
    ax.arrow(*F["c"], *(F["ud"]*F["hu"]), color="red", width=2)
    ax.arrow(*F["c"], *(F["vd"]*F["hv"]), color="lime", width=2)
    ax.set_title(f"{date} eye frame (red=u, green=v)"); ax.set_xticks([]); ax.set_yticks([]); plt.show()

# ---------------------------------------------------------------- QC & analysis plots
def plot_pupil_xy(date, n=1000, high=50.0):
    """Pupil x and y over the session for n equally-spaced frames, both trackers."""
    r = track_both(session_for_date(date), n, high)
    if r["n"] == 0:
        raise ValueError("no pupil detected for " + date)
    tm = r["t"] / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(tm, r["ell"][:, 0], ".", ms=3, c="tab:cyan", label="robust ellipse")
    ax[0].plot(tm, r["onl"][:, 0], ".", ms=3, c="tab:orange", label="online centroid")
    ax[0].set_ylabel("pupil x (px)"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"{date}  {r['n']}/{n} frames with pupil")
    ax[1].plot(tm, r["ell"][:, 1], ".", ms=3, c="tab:cyan")
    ax[1].plot(tm, r["onl"][:, 1], ".", ms=3, c="tab:orange")
    ax[1].set_ylabel("pupil y (px)"); ax[1].set_xlabel("time (min)")
    plt.tight_layout(); plt.show()
    return r

def tracker_agreement(dates, n=120, high=50.0):
    """Scatter the two trackers against each other per frame: ellipse-x vs online-x and
    ellipse-y vs online-y (dashed = identity). Tight scatter on the line = the online
    centroid faithfully follows the pupil."""
    if isinstance(dates, str):
        dates = [dates]
    data = [(d, track_both(session_for_date(d), n, high)) for d in dates]
    data = [(d, r) for d, r in data if r["n"] > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for j, lab in enumerate(["x", "y"]):
        vals = []
        for d, r in data:
            e = r["ell"][:, j]; o = r["onl"][:, j]
            ax[j].scatter(e, o, s=10, alpha=0.5, label=d); vals += list(e) + list(o)
        if vals:
            lo, hi = min(vals), max(vals); ax[j].plot([lo, hi], [lo, hi], "k--", lw=1)
        ax[j].set_xlabel(f"ellipse {lab} (px)"); ax[j].set_ylabel(f"online {lab} (px)")
        ax[j].set_title(f"pupil {lab}: robust vs online"); ax[j].set_aspect("equal", "box")
        ax[j].legend(fontsize=7)
    plt.tight_layout(); plt.show()
    for d, r in data:
        rx = np.corrcoef(r["ell"][:, 0], r["onl"][:, 0])[0, 1]
        ry = np.corrcoef(r["ell"][:, 1], r["onl"][:, 1])[0, 1]
        mdx = np.median(np.abs(r["ell"][:, 0] - r["onl"][:, 0]))
        mdy = np.median(np.abs(r["ell"][:, 1] - r["onl"][:, 1]))
        print(f"{d}: corr x={rx:.3f} y={ry:.3f}  median|Δx|={mdx:.1f}px |Δy|={mdy:.1f}px  n={r['n']}")

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

def _eyeframe_by_session(dates, axis="u", tracker="ell", n=200, high=50.0, landmarks=None):
    """Per session: (summary-agnostic) 1-D array of eye-frame coord `axis` for `tracker`."""
    LM = landmarks if landmarks is not None else LANDMARKS
    j = 0 if axis == "u" else 1
    out = []
    for d in dates:
        if d not in LM:
            raise ValueError(f"no landmarks for {d} - clicker_if_missing('{d}') first")
        r = track_both(session_for_date(d), n, high); F = eye_frame(LM[d])
        uv = np.array([to_eye(p, F) for p in r[tracker]]).reshape(-1, 2)
        out.append(uv[:, j])
    return out   # list of arrays, one per session

def _perm_statistic(A_sessions, B_sessions, stat, summary):
    """Statistic between two groups of per-session arrays."""
    agg = np.mean if summary == "mean" else np.median
    if stat in ("mean_diff", "median_diff", "var_diff"):
        a = np.array([agg(s) for s in A_sessions]); b = np.array([agg(s) for s in B_sessions])
        if stat == "var_diff":
            return abs(np.var(a, ddof=1) - np.var(b, ddof=1))
        return abs(np.mean(a) - np.mean(b)) if stat == "mean_diff" else abs(np.median(a) - np.median(b))
    # distribution-distance stats on pooled frames (session-level shuffle preserved by caller)
    A = np.concatenate(A_sessions); B = np.concatenate(B_sessions)
    from scipy import stats as ss
    if stat == "ks":
        return ss.ks_2samp(A, B).statistic
    if stat == "wasserstein":
        return ss.wasserstein_distance(A, B)
    raise ValueError("unknown stat " + stat)

def permutation_test(centered=None, biased=None, n=200, high=50.0, summary="mean",
                     stat="mean_diff", n_perm=20000, landmarks=None, plot=True):
    """Two-condition label-shuffle test, for both trackers and both axes.

    The exchangeable unit is the SESSION: we permute which sessions are 'centered' vs 'biased'
    (exact enumeration of all C(N, n_centered) splits when feasible, else `n_perm` random draws),
    recompute `stat`, and compare the observed value to that null.

    stat: 'mean_diff' | 'median_diff' | 'var_diff' (on per-session summaries) or
          'ks' | 'wasserstein' (distribution distance on pooled frames). Two-sided by
          construction (all these statistics are non-negative distances).
    """
    from itertools import combinations
    trackers = [("ell", "robust ellipse"), ("onl", "online centroid")]
    axes = [("u", "horizontal (x)"), ("v", "vertical (y)")]
    N = len(centered) + len(biased); nc = len(centered); dates = list(centered) + list(biased)
    idx = np.arange(N)
    combos = list(combinations(range(N), nc))
    exact = len(combos) <= n_perm
    if not exact:
        combos = None
    fig = None
    if plot:
        fig, axfig = plt.subplots(2, 2, figsize=(13, 9))
    results = {}
    for i, (tr, tlab) in enumerate(trackers):
        for k, (ax, alab) in enumerate(axes):
            sess = _eyeframe_by_session(dates, ax, tr, n, high, landmarks)
            obs = _perm_statistic(sess[:nc], sess[nc:], stat, summary)
            null = []
            if exact:
                for comb in combos:
                    m = np.zeros(N, bool); m[list(comb)] = True
                    null.append(_perm_statistic([sess[a] for a in idx[m]], [sess[a] for a in idx[~m]], stat, summary))
            else:
                for _ in range(n_perm):
                    m = np.zeros(N, bool); m[np.random.choice(N, nc, replace=False)] = True
                    null.append(_perm_statistic([sess[a] for a in idx[m]], [sess[a] for a in idx[~m]], stat, summary))
            null = np.array(null)
            p = (np.sum(null >= obs - 1e-12) + (0 if exact else 1)) / (len(null) + (0 if exact else 1))
            results[(tr, ax)] = dict(observed=float(obs), p=float(p), n_perm=len(null), exact=exact)
            if plot:
                a = axfig[i][k]
                a.hist(null, bins=40, color="0.7"); a.axvline(obs, color="red", lw=2)
                a.set_title(f"{tlab} / {alab}\n{stat}: obs={obs:.3f}, p={p:.3f}"
                            f" ({'exact '+str(len(null)) if exact else str(len(null))+' rand'})")
                a.set_xlabel(f"{stat} under shuffled labels"); a.set_ylabel("count")
    if plot:
        plt.tight_layout(); plt.show()
    print(f"permutation test  stat={stat}  summary={summary}  "
          f"({'exact' if exact else 'random'} null, centered n={nc}, biased n={N-nc})")
    for (tr, ax), v in results.items():
        print(f"  {tr}/{ax}: observed={v['observed']:.4f}  p={v['p']:.4f}  (n_perm={v['n_perm']})")
    return results

def _group_test(groups):
    """groups: list of 1-D arrays. Two -> Welch t-test; >2 -> one-way ANOVA.
    Returns (label, stat, p) or (label, nan, nan) if any group has <2 points."""
    from scipy import stats
    groups = [np.asarray(g, float) for g in groups]
    if any(len(g) < 2 for g in groups):
        return ("t-test" if len(groups) == 2 else "ANOVA"), float("nan"), float("nan")
    if len(groups) == 2:
        s, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        return "Welch t-test", float(s), float(p)
    s, p = stats.f_oneway(*groups)
    return "one-way ANOVA", float(s), float(p)

def compare_agreement(centered=None, biased=None, conditions=None, n=120, high=50.0, bins=50):
    """Per condition, compare the two trackers directly: ellipse-x vs online-x and
    ellipse-y vs online-y scatter (dashed = identity), plus the discrepancy
    delta = online - ellipse. If delta differs between conditions, a tracker artifact
    (not real gaze) could explain a group difference. No landmarks needed."""
    if conditions is None:
        conditions = {"centered": centered or [], "biased": biased or []}
    names = list(conditions)
    palette = ["tab:green", "tab:red", "tab:blue", "tab:purple", "tab:orange"]
    colors = {nm: palette[i % len(palette)] for i, nm in enumerate(names)}
    pf = {nm: {"ex": [], "ox": [], "ey": [], "oy": []} for nm in names}   # per-frame
    dday = {nm: {"dx": [], "dy": []} for nm in names}                     # per-session median delta
    for nm in names:
        for d in conditions[nm]:
            r = track_both(session_for_date(d), n, high)
            if r["n"] == 0:
                continue
            ex, ox = r["ell"][:, 0], r["onl"][:, 0]; ey, oy = r["ell"][:, 1], r["onl"][:, 1]
            pf[nm]["ex"] += list(ex); pf[nm]["ox"] += list(ox)
            pf[nm]["ey"] += list(ey); pf[nm]["oy"] += list(oy)
            dday[nm]["dx"].append(float(np.median(ox - ex))); dday[nm]["dy"].append(float(np.median(oy - ey)))
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for axis, (a, b, lab) in zip([ax[0, 0], ax[0, 1]], [("ex", "ox", "x"), ("ey", "oy", "y")]):
        vals = []
        for nm in names:
            axis.scatter(pf[nm][a], pf[nm][b], s=8, alpha=0.4, color=colors[nm], label=nm)
            vals += pf[nm][a] + pf[nm][b]
        if vals:
            lo, hi = min(vals), max(vals); axis.plot([lo, hi], [lo, hi], "k--", lw=1)
        axis.set_xlabel(f"ellipse {lab} (px)"); axis.set_ylabel(f"online {lab} (px)")
        axis.set_title(f"pupil {lab}: robust vs online"); axis.set_aspect("equal", "box"); axis.legend(fontsize=8)
    for axis, key, lab in [(ax[1, 0], ("ox", "ex"), "x"), (ax[1, 1], ("oy", "ey"), "y")]:
        alld = []
        for nm in names:
            dd = np.array(pf[nm][key[0]]) - np.array(pf[nm][key[1]]); alld += list(dd)
        rng = (np.percentile(alld, 1), np.percentile(alld, 99)) if alld else (-1, 1)
        for nm in names:
            dd = np.array(pf[nm][key[0]]) - np.array(pf[nm][key[1]])
            axis.hist(dd, bins=bins, range=rng, alpha=0.5, color=colors[nm], density=True, label=nm)
        axis.axvline(0, color="k", ls=":"); axis.set_xlabel(f"delta {lab} = online - ellipse (px)")
        axis.set_ylabel("density"); axis.set_title(f"tracker discrepancy in {lab}"); axis.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    for lab, keys in [("delta x", "dx"), ("delta y", "dy")]:
        for nm in names:
            arr = np.array(dday[nm][keys])
            print(f"{lab} {nm:<9} per-session median: {np.round(arr,2).tolist()}  mean={np.mean(arr):+.2f}px" if len(arr) else f"{lab} {nm}: (none)")
        ls, st, p = _group_test([dday[nm][keys] for nm in names])
        print(f"[{lab}] session-level {ls} (n_days={[len(conditions[nm]) for nm in names]}): stat={st:.3f} p={p:.4g}"
              + ("  <-- need >=2 days/condition" if p != p else ""))
    return dict(perframe=pf, daymedian=dday)

def compare_conditions(centered=None, biased=None, conditions=None,
                       n=120, high=50.0, bins=60, summary="mean", landmarks=None):
    """Compare eye-frame pupil position between conditions, for both trackers and both axes
    (u = horizontal / eye-frame x, v = vertical / eye-frame y).

    Provide either centered=[...], biased=[...] (two groups) or
    conditions={'name': [dates], ...} for two or more groups.
    The test asks whether the MEAN eye-frame position differs across conditions, run on one
    summary value per session (summary='mean' or 'median'; independent unit = the session,
    n = #days): two groups -> Welch t-test, >2 -> one-way ANOVA. A frame-level test is also
    printed but is pseudo-replicated (correlated frames) and only illustrative.
    """
    LM = landmarks if landmarks is not None else LANDMARKS
    agg = np.mean if summary == "mean" else np.median
    if conditions is None:
        conditions = {"centered": centered or [], "biased": biased or []}
    names = list(conditions)
    palette = ["black", "red", "tab:blue", "tab:purple", "tab:orange"]
    colors = {nm: palette[i % len(palette)] for i, nm in enumerate(names)}
    trackers = [("ell", "robust ellipse"), ("onl", "online centroid")]
    AX = [("u", 0, "horizontal (eye-frame x)"), ("v", 1, "vertical (eye-frame y)")]
    # per-frame coords and per-session summaries, per condition/tracker/axis
    pooled = {nm: {tr: {"u": [], "v": []} for tr, _ in trackers} for nm in names}
    daysum = {nm: {tr: {"u": [], "v": []} for tr, _ in trackers} for nm in names}
    perday = []
    for nm in names:
        for d in conditions[nm]:
            if d not in LM:
                raise ValueError(f"no landmarks for {d} - clicker_if_missing('{d}') first")
            r = track_both(session_for_date(d), n, high); F = eye_frame(LM[d])
            row = [nm, d]
            for tr, _ in trackers:
                uv = np.array([to_eye(p, F) for p in r[tr]]).reshape(-1, 2)
                pooled[nm][tr]["u"] += list(uv[:, 0]); pooled[nm][tr]["v"] += list(uv[:, 1])
                su, sv = float(agg(uv[:, 0])), float(agg(uv[:, 1]))
                daysum[nm][tr]["u"].append(su); daysum[nm][tr]["v"].append(sv)
                row += [su, sv]
            perday.append(tuple(row + [r["n"]]))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for i, (tr, tlab) in enumerate(trackers):
        for ax_key, j, alab in AX:
            ax[i][j].axvline(0, color="0.6", ls=":", lw=1)      # eye-center reference
            for nm in names:
                ax[i][j].hist(pooled[nm][tr][ax_key], bins=bins, range=(-1, 1), histtype="step",
                              lw=1.6, color=colors[nm], label=nm, density=True)
                for sv in daysum[nm][tr][ax_key]:               # per-session mean
                    ax[i][j].axvline(sv, color=colors[nm], lw=0.7, alpha=0.55)
            ax[i][j].set_title(f"{tlab}: {alab}"); ax[i][j].legend(fontsize=8)
            ax[i][j].set_xlabel("pupil position in eye frame (-1 .. +1, 0 = centered)")
            ax[i][j].set_ylabel("density")
    plt.tight_layout(); plt.show()

    print(f"per-session {summary} of eye-frame position (u=horizontal, v=vertical):")
    print(f"{'cond':<9}{'date':<12}{'u_rob':>7}{'v_rob':>7}{'u_onl':>7}{'v_onl':>7}{'n':>6}")
    for r in perday:
        print(f"{r[0]:<9}{r[1]:<12}{r[2]:>7.3f}{r[3]:>7.3f}{r[4]:>7.3f}{r[5]:>7.3f}{r[6]:>6d}")
    for tr, tlab in trackers:
        for ax_key, _, alab in AX:
            for nm in names:
                vals = daysum[nm][tr][ax_key]
                m = np.mean(vals) if vals else float("nan")
                print(f"  {tlab:<16} {alab:<24} {nm:<9} mean of session-{summary} = {m:+.3f}  days={len(vals)}")
            ls, lstat, lp = _group_test([daysum[nm][tr][ax_key] for nm in names])   # session-level (valid)
            fs, fstat, fp = _group_test([pooled[nm][tr][ax_key] for nm in names])   # frame-level (illustrative)
            print(f"  -> [{tlab}/{alab}] session-level {ls} (n_days={[len(conditions[nm]) for nm in names]}): "
                  f"stat={lstat:.3f} p={lp:.4g}" + ("  <-- need >=2 days/condition" if lp != lp else ""))
            print(f"     [{tlab}/{alab}] frame-level {fs} (pseudo-replicated, illustrative): "
                  f"stat={fstat:.3f} p={fp:.4g}\n")
    return dict(pooled=pooled, daysum=daysum, perday=perday)
