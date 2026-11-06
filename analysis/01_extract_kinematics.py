"""
Step 1 - Computer vision: extract collapse kinematics from the eyewitness video.

Method
  * Pyramidal Lucas-Kanade optical flow with forward-backward consistency check.
  * Camera is hand-held and zooms/pans, so every measurement is made *relative*
    to something:
      - early phase (0-2.3 s): facade patches vs a static boundary wall at ground
        level, normalised by a ruler on the facade itself (distance between two
        facade patches ~6 storeys apart) -> zoom-invariant.
      - left-vs-centre: two patches at the same level of the same facade
        -> zoom/pan-invariant measure of differential settlement (tilt).
      - roof phase (3.7-5.0 s): white end-facade features vs the static street /
        background, with a similarity transform estimated from static points.
  * Pixels -> metres through the storey pitch measured on the facade
    (autocorrelation of the slab-line pattern) and an assumed storey height.

Output: kinematics.csv  (t [s], z [m, downward], segment, sigma [m])
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path(__file__).resolve().parent.parent / "VID-20261023-WA0009.mp4"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

STOREY_H = 3.0          # m, assumed floor-to-floor height (typical; +/-10 %)
LK = dict(winSize=(15, 15), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def load_frames():
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return fps, frames


def box_mask(shape, rects):
    m = np.zeros(shape, np.uint8)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = 255
    return m


def track_patches(frames, a, b, patches):
    """Median position of each patch from frame a to b (NaN once lost)."""
    f0 = frames[a]
    pts, ok, pos = {}, {}, {}
    for k, r in patches.items():
        pts[k] = cv2.goodFeaturesToTrack(f0, 80, 0.005, 3, mask=box_mask(f0.shape, [r]))
        ok[k] = np.ones(len(pts[k]), bool)
        pos[k] = [np.median(pts[k].reshape(-1, 2), 0)]
    for i in range(a + 1, b):
        prev, cur = frames[i - 1], frames[i]
        for k in patches:
            n, s, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts[k], None, **LK)
            bk, _, _ = cv2.calcOpticalFlowPyrLK(cur, prev, n, None, **LK)
            fb = np.linalg.norm((bk - pts[k]).reshape(-1, 2), axis=1)
            ok[k] &= (s.ravel() == 1) & (fb < 0.7)
            disp = (n - pts[k]).reshape(-1, 2)[ok[k]]
            pos[k].append(pos[k][-1] + np.median(disp, 0) if ok[k].sum() >= 3
                          else np.array([np.nan, np.nan]))
            pts[k] = n
    return {k: np.array(v) for k, v in pos.items()}


def track_vs_static(frames, a, b, moving_rect, static_rects):
    """Median vertical displacement of moving features, in frame-a pixel units,
    after removing camera motion estimated (similarity) from static features."""
    f0 = frames[a]
    mov = cv2.goodFeaturesToTrack(f0, 300, 0.005, 4, mask=box_mask(f0.shape, [moving_rect]))
    sta = cv2.goodFeaturesToTrack(f0, 300, 0.01, 5, mask=box_mask(f0.shape, static_rects))
    pm, ps = mov.copy(), sta.copy()
    okm, oks = np.ones(len(pm), bool), np.ones(len(ps), bool)
    A = np.eye(3)
    out = [(a, 0.0)]
    for i in range(a + 1, b):
        prev, cur = frames[i - 1], frames[i]
        nm, sm, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pm, None, **LK)
        bm, _, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nm, None, **LK)
        okm &= (sm.ravel() == 1) & (np.linalg.norm((bm - pm).reshape(-1, 2), axis=1) < 0.7)
        ns, ss, _ = cv2.calcOpticalFlowPyrLK(prev, cur, ps, None, **LK)
        bs, _, _ = cv2.calcOpticalFlowPyrLK(cur, prev, ns, None, **LK)
        oks &= (ss.ravel() == 1) & (np.linalg.norm((bs - ps).reshape(-1, 2), axis=1) < 0.7)
        if oks.sum() < 8 or okm.sum() < 8:
            break
        M, _ = cv2.estimateAffinePartial2D(ns[oks], ps[oks], method=cv2.RANSAC,
                                           ransacReprojThreshold=1.5)
        A = A @ np.vstack([M, [0, 0, 1]])
        q = (A @ np.c_[nm[okm].reshape(-1, 2), np.ones(okm.sum())].T).T[:, :2]
        out.append((i, float(np.median(q[:, 1] - mov[okm].reshape(-1, 2)[:, 1]))))
        pm, ps = nm, ns
    return out


def storey_pitch(img, x0, x1, y0, y1):
    p = img[y0:y1, x0:x1].astype(float).mean(1)
    p -= p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac /= ac[0]
    peaks = [i for i in range(15, 90) if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]]
    return max(peaks[:2], key=lambda i: ac[i])


def main():
    fps, frames = load_frames()
    print(f"{len(frames)} frames @ {fps:.2f} fps, {frames[0].shape[::-1]}")

    # ---------- Phase A: initiation, frames 0-70 (camera on building base) ----------
    P = {"U": (262, 238, 312, 270),    # barred window ~6 storeys above L
         "U2": (300, 180, 372, 240),   # upper pier
         "M": (318, 380, 372, 425),    # pier ~2 storeys above 1st suspended slab
         "L": (255, 444, 307, 472),    # openings just above 1st suspended slab, centre
         "LL": (140, 440, 200, 478),   # same level, left (street-corner) end
         "G": (300, 522, 378, 556)}    # static boundary wall at ground level
    pos = track_patches(frames, 0, 75, P)
    span_UL = pos["L"][:, 1] - pos["U"][:, 1]
    px_per_storey_A = span_UL[0] / 6.0                     # ruler: U-L = 6 storeys
    pitch_check = storey_pitch(frames[0], 320, 375, 0, 300)
    print(f"phase A: {px_per_storey_A:.1f} px/storey (U-L ruler), "
          f"autocorr pitch upper facade {pitch_check} px")
    mA = STOREY_H / px_per_storey_A

    sUL = span_UL / span_UL[0]
    sM = (pos["M"][:, 1] - pos["U2"][:, 1]) / (pos["M"][0, 1] - pos["U2"][0, 1])
    z_L = ((pos["G"][0, 1] - pos["L"][0, 1]) - (pos["G"][:, 1] - pos["L"][:, 1]) / sUL) * mA
    z_M = ((pos["G"][0, 1] - pos["M"][0, 1]) - (pos["G"][:, 1] - pos["M"][:, 1]) / sM) * mA
    tilt = ((pos["LL"][:, 1] - pos["L"][:, 1]) / sUL - (pos["LL"][0, 1] - pos["L"][0, 1])) * mA

    # ---------- Phase B: roof descent, frames 112-150 (camera re-framed) ----------
    rb = track_vs_static(frames, 112, 150, (133, 110, 190, 420),
                         [(140, 700, 480, 864), (88, 260, 128, 420)])
    px_per_storey_B = storey_pitch(frames[112], 140, 180, 120, 500)
    mB = STOREY_H / px_per_storey_B
    print(f"phase B: {px_per_storey_B} px/storey (white end facade)")

    rows = []
    for i in range(len(z_L)):
        t = i / fps
        if i <= 28 and np.isfinite(z_L[i]):                       # pre-zoom, centre at rest?
            rows.append((t, z_L[i], "A_centre_L", 0.15))
        if 45 <= i <= 68 and np.isfinite(z_M[i]):                 # zoom finished, ruler stable
            rows.append((t, z_M[i], "A_centre_M", 0.10))
        if np.isfinite(tilt[i]):
            rows.append((t, tilt[i], "A_left_minus_centre", 0.10))
    for i, dy in rb:
        rows.append((i / fps, dy * mB, "B_roof", 0.25))

    with open(OUT / "kinematics.csv", "w") as f:
        f.write("t_s,z_m,segment,sigma_m\n")
        for r in rows:
            f.write(f"{r[0]:.4f},{r[1]:.4f},{r[2]},{r[3]}\n")
    json.dump(dict(fps=fps, storey_h=STOREY_H, px_per_storey_A=px_per_storey_A,
                   px_per_storey_B=float(px_per_storey_B),
                   zoom_A=sUL.tolist()), open(OUT / "kinematics_meta.json", "w"), indent=1)
    print(f"wrote {len(rows)} rows -> {OUT/'kinematics.csv'}")


if __name__ == "__main__":
    sys.exit(main())
