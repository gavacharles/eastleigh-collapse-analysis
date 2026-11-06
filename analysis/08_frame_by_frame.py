"""
Step 8 - Frame-by-frame forensic record (image + sound).

For every one of the 298 frames:
  * tracked patches drawn on the image (phase A), with measured descent and
    left-end drop, model z / v / resistance, phase label
  * audio: RMS level (dB) and spectral centroid from the clip's AAC track,
    synchronised to the frame
Writes
  frames/f###.jpg               annotated frames
  out/frame_metrics.csv         one row per frame
  out/fig_audio_timeline.png    spectrogram + level + kinematics on one time axis
  out/frame_by_frame.mp4        annotated video
  frame_viewer.html             local viewer: slider, arrow-key stepping, metrics
The footage shows identifiable people: keep these outputs local.
"""
import importlib.util
import json
import subprocess
import wave
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit import CREDIT, SRC_CV, SRC_MODEL, SRC_VIDEO, footer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FR = HERE / "frames"
FR.mkdir(exist_ok=True)
VIDEO = HERE.parent / "VID-20261023-WA0009.mp4"

spec = importlib.util.spec_from_file_location("kin", HERE / "01_extract_kinematics.py")
kin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kin)


# ------------------------------------------------------------------ audio
def load_audio():
    wav = OUT / "audio.wav"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(VIDEO),
                    "-ac", "1", "-ar", "22050", str(wav)], check=True)
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
    return sr, a


def audio_features(sr, a, fps, nframes):
    hop = sr / fps
    rms_db, cent = [], []
    for i in range(nframes):
        s = a[int(i * hop):int((i + 1) * hop)]
        if len(s) < 16:
            rms_db.append(np.nan); cent.append(np.nan); continue
        rms_db.append(20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9))
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s))))
        f = np.fft.rfftfreq(len(s), 1 / sr)
        cent.append(float((sp * f).sum() / (sp.sum() + 1e-12)))
    return np.array(rms_db), np.array(cent)


# ------------------------------------------------------------------ main
def main():
    fps, gray = kin.load_frames()
    cap = cv2.VideoCapture(str(VIDEO))
    color = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        color.append(f)
    n = len(color)
    P = {"U": (262, 238, 312, 270), "L": (255, 444, 307, 472),
         "LL": (140, 440, 200, 478), "G": (300, 522, 378, 556)}
    pos = kin.track_patches(gray, 0, 75, P)
    sr, a = load_audio()
    rms, cent = audio_features(sr, a, fps, n)

    tl = json.load(open(OUT / "animation_timeline.json"))
    rows = np.array([[r[0], r[1], r[2], np.nan if r[3] is None else r[3], r[4]] for r in tl["rows"]])
    raw = np.genfromtxt(OUT / "kinematics.csv", delimiter=",", names=True, dtype=None, encoding=None)
    tilt = raw[raw["segment"] == "A_left_minus_centre"]
    t0 = tl["t0"]

    def phase(t, z):
        if t < 0.4: return "LOCAL FAILURE AT L0 CORNER"
        if t < t0: return "ZIPPER ALONG L0"
        if t < 2.3: return "GLOBAL ONSET - BLOCK DROPS"
        if t < 3.75: return "CAMERA OFF BUILDING (model bridge)"
        if t < 5.5: return "CRUSH-UP FROM BASE"
        return "DUST CLOUD - STRUCTURE OBSCURED"

    metrics = []
    writer = imageio.get_writer(OUT / "frame_by_frame.mp4", fps=fps, codec="libx264", quality=7,
                                macro_block_size=1)
    db_lo, db_hi = np.nanpercentile(rms, 2), np.nanmax(rms)
    for i in range(n):
        t = i / fps
        z, v, rho = (np.interp(t, rows[:, 0], rows[:, k]) for k in (1, 2, 3))
        drop = float(np.interp(t, tilt["t_s"], tilt["z_m"])) if t <= tilt["t_s"][-1] else np.nan
        im = color[i].copy()
        if i < len(pos["U"]):
            for k, (x0, y0, x1, y1) in P.items():
                p = pos[k][i]
                if np.isfinite(p).all():
                    p0 = pos[k][0]
                    dx, dy = p - p0
                    c = (0, 200, 255) if k != "G" else (120, 255, 120)
                    cv2.rectangle(im, (int(x0 + dx), int(y0 + dy)), (int(x1 + dx), int(y1 + dy)), c, 1)
                    cv2.putText(im, k, (int(x0 + dx), int(y0 + dy) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        # HUD
        hud = np.zeros((190, im.shape[1], 3), np.uint8)
        put = lambda s, y, c=(255, 255, 255), sc=0.48: cv2.putText(hud, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)
        put(f"frame {i:3d}   t = {t:5.2f} s", 20, (0, 220, 255), 0.55)
        put(phase(t, z), 42, (80, 80, 255), 0.5)
        put(f"model: descent {z:5.1f} m   v {v:4.1f} m/s   R/W {'-' if np.isnan(rho) else f'{rho:.2f}'}", 66)
        put(f"measured left-end drop: {'-' if np.isnan(drop) else f'{drop:.2f} m'}", 88)
        put(f"audio {rms[i]:5.1f} dBFS   centroid {cent[i]:5.0f} Hz", 110)
        lvl = np.clip((rms[i] - db_lo) / (db_hi - db_lo), 0, 1) if np.isfinite(rms[i]) else 0
        cv2.rectangle(hud, (8, 122), (8 + int(lvl * (im.shape[1] - 16)), 138), (0, 170, 255), -1)
        cv2.rectangle(hud, (8, 122), (im.shape[1] - 8, 138), (90, 90, 90), 1)
        put("Data: eyewitness video 23 Oct 2026; CV tracking,", 158, (170, 170, 170), 0.4)
        put("crush-up model (this analysis) | " + CREDIT, 176, (170, 170, 170), 0.4)
        frame = np.vstack([im, hud])
        cv2.imwrite(str(FR / f"f{i:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        metrics.append(dict(frame=i, t=round(t, 3), phase=phase(t, z), z_model=round(float(z), 3),
                            v_model=round(float(v), 3), rho_model=None if np.isnan(rho) else round(float(rho), 3),
                            left_drop_meas=None if np.isnan(drop) else round(drop, 3),
                            audio_db=round(float(rms[i]), 2), audio_centroid_hz=round(float(cent[i]), 0)))
    writer.close()
    keys = list(metrics[0])
    with open(OUT / "frame_metrics.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for m in metrics:
            f.write(",".join("" if m[k] is None else str(m[k]) for k in keys) + "\n")

    # ---------------- audio + kinematics figure
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
    ax[0].specgram(a, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-120)
    ax[0].set(ylim=(0, 6000), ylabel="frequency [Hz]", title="Sound of the collapse (clip audio track)")
    tt = np.arange(n) / fps
    ax[1].plot(tt, rms, color="#c77700")
    ax[1].set(ylabel="level [dBFS]")
    ax[2].plot(rows[:, 0], rows[:, 2], color="#1f6aa5", label="model velocity [m/s]")
    ax[2].plot(tilt["t_s"], np.array(tilt["z_m"]) * 10, color="#c62f2f", label="measured left-end drop x10 [m]")
    ax[2].set(xlabel="video time [s]", ylabel="value")
    ax[2].legend(fontsize=8)
    for a_ in ax:
        for x, lab in ((0.4, "zipper"), (t0, "global onset"), (2.3, "camera off"), (3.75, "camera back")):
            a_.axvline(x, color="0.4", lw=0.8, ls="--")
        a_.set_xlim(0, n / fps)
    for x, lab in ((0.4, "zipper"), (t0, "onset"), (2.3, "cam off"), (3.75, "cam back")):
        ax[0].text(x + 0.03, 5600, lab, color="w", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    footer(fig, SRC_VIDEO + " audio track (AAC)", SRC_CV, SRC_MODEL)
    fig.savefig(OUT / "fig_audio_timeline.png", dpi=140)

    # ---------------- local viewer
    html = (HERE / "frame_viewer_template.html").read_text().replace("/*METRICS*/[]", json.dumps(metrics))
    (HERE / "frame_viewer.html").write_text(html)
    print(f"{n} frames annotated; audio dB range {np.nanmin(rms):.1f}..{np.nanmax(rms):.1f}")
    # simple onset report: frames where level jumps > 6 dB over the previous 0.3 s median
    med = np.array([np.nanmedian(rms[max(0, i - 9):i]) if i > 2 else rms[i] for i in range(n)])
    jumps = [(i, round(i / fps, 2), round(rms[i] - med[i], 1)) for i in range(3, n)
             if med[i] > -90 and rms[i] - med[i] > 6]       # skip the silent lead-in
    print("audio onsets (frame, t, +dB):", jumps[:15])
    json.dump(dict(audio_onsets=jumps), open(OUT / "audio_onsets.json", "w"))


if __name__ == "__main__":
    main()
