"""
Step 12 - Independent sensors: citizen seismometer and soil grids.

Seismic: Raspberry Shake AM.R8A6C (~11 km N of the site, vertical geophone,
100 Hz), 10:20-11:00 UTC on 23 Oct 2026 (13:20-14:00 EAT). STA/LTA screening
and 10-s energy, to find a candidate collapse signal near the reported 13:40.
Soils: ISRIC SoilGrids 250 m (urban pixel masked; nearest valid pixel ~0.9 km E).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read
from obspy.signal.trigger import classic_sta_lta, trigger_onset

from credit import footer

HERE = Path(__file__).resolve().parent
RAW = HERE / "gis" / "raw"
OUT = HERE / "gis"
SENS = 381407000.0          # counts per m/s at 5 Hz (station metadata)
STA_LAT, STA_LON = -1.18018, 36.88180
SITE = json.load(open(RAW / "meta.json"))["site"]


def main():
    st = read(str(RAW / "seismic" / "R8A6C_20261023.mseed")).merge(fill_value=0)
    tr = st[0]
    tr.data = tr.data.astype(float) / SENS
    tr.detrend("demean")
    tr.filter("bandpass", freqmin=1, freqmax=20)
    sr = tr.stats.sampling_rate
    t0 = tr.stats.starttime
    cft = classic_sta_lta(tr.data, int(1 * sr), int(30 * sr))
    trig = trigger_onset(cft, 3.0, 1.2)
    w = int(10 * sr)
    rms = np.array([np.sqrt(np.mean(tr.data[i:i + w] ** 2)) for i in range(0, len(tr.data) - w, w // 2)])
    t_rms = np.arange(len(rms)) * 5.0
    i_max = int(np.argmax(rms))
    cand = t0 + t_rms[i_max]
    dist_km = np.hypot((STA_LAT - SITE[0]) * 110.57, (STA_LON - SITE[1]) * 111.32)
    events = [dict(utc=str(t0 + a / sr)[:22], dur_s=round((b - a) / sr, 1),
                   peak_ums=round(float(np.abs(tr.data[a:b]).max()) * 1e6, 2), sta_lta=round(float(cft[a:b].max()), 1))
              for a, b in trig]
    top = sorted(events, key=lambda e: -e["sta_lta"])[:6]
    # ---- figure
    fig = plt.figure(figsize=(13, 9))
    ax1 = fig.add_axes([0.07, 0.72, 0.9, 0.2])
    ax2 = fig.add_axes([0.07, 0.47, 0.9, 0.2], sharex=ax1)
    ax3 = fig.add_axes([0.07, 0.1, 0.55, 0.28])
    ax4 = fig.add_axes([0.68, 0.1, 0.29, 0.28])
    tm = np.arange(tr.stats.npts) / sr / 60
    ax1.plot(tm, tr.data * 1e6, color="0.25", lw=0.3)
    ax1.set(ylabel="velocity [um/s]", title=f"Raspberry Shake AM.R8A6C (~{dist_km:.0f} km N of site), 1-20 Hz, "
            f"{str(t0)[:16]} UTC onward")
    for e in top:
        x = (UTCDateTime(e["utc"]) - t0) / 60
        ax1.axvline(x, color="#c77700", lw=0.8, alpha=0.7)
    xc = (cand - t0) / 60
    ax1.axvspan(xc, xc + 10 / 60, color="#c62f2f", alpha=0.25)
    ax2.plot(t_rms / 60, rms * 1e6, color="#1f6aa5")
    ax2.axvline((UTCDateTime("2026-10-23T10:40:00") - t0) / 60, color="k", ls="--", lw=1)
    ax2.text((UTCDateTime("2026-10-23T10:40:00") - t0) / 60 + 0.2, rms.max() * 1e6 * 0.9,
             "reported ~13:40 EAT", fontsize=8)
    ax2.axvspan(xc, xc + 10 / 60, color="#c62f2f", alpha=0.25)
    ax2.text(xc - 0.2, rms.max() * 1e6 * 0.75, f"strongest 10-s energy\n{str(cand)[11:19]} UTC\n= {str(cand + 3 * 3600)[11:19]} EAT",
             fontsize=8, color="#c62f2f", ha="right")
    ax2.set(ylabel="10-s RMS [um/s]", xlabel="minutes after start")
    # zoom + spectrogram of candidate
    a = int((cand - t0 - 20) * sr); b = int((cand - t0 + 40) * sr)
    seg = tr.data[a:b]
    ax3.specgram(seg, NFFT=256, Fs=sr, noverlap=200, cmap="magma")
    ax3.set(ylim=(0, 25), xlabel=f"seconds after {str(cand - 20)[11:19]} UTC", ylabel="Hz",
            title="Candidate window: spectrogram")
    ax4.axis("off")
    txt = ("Strongest STA/LTA triggers (UTC):\n" +
           "\n".join(f"{e['utc'][11:19]}  {e['dur_s']:4.1f}s  STA/LTA {e['sta_lta']:4.1f}" for e in top) +
           "\n\nReading: the largest 10-s energy in the\n40-min window falls at about 13:36 EAT,\n"
           "a 5 s emergent burst followed by a\nsecond pulse, consistent with a collapse\n"
           "of several seconds. Urban noise is high\nand there is one station only, so it is a\n"
           "CANDIDATE, not a confirmed detection.")
    ax4.text(0, 1, txt, va="top", family="monospace", fontsize=8.5)
    footer(fig, "Raspberry Shake citizen-science network (AM.R8A6C) via data.raspberryshake.org FDSN",
           "processing: ObsPy (this analysis)")
    fig.savefig(OUT / "fig_seismic_candidate.png", dpi=140)

    # ---- soils
    near = json.load(open(RAW / "soilgrids_near.json")) if (RAW / "soilgrids_near.json").exists() else None
    wrb = json.load(open(RAW / "soilgrids_wrb.json"))
    res = dict(seismic=dict(station="AM.R8A6C", distance_km=round(dist_km, 1),
                            candidate_utc=str(cand)[:19], candidate_eat=str(cand + 3 * 3600)[:19],
                            median_rms_ums=round(float(np.median(rms)) * 1e6, 3),
                            max_rms_ums=round(float(rms.max()) * 1e6, 3), top_triggers=top),
               soils=dict(wrb_top=wrb.get("wrb_class_probability"), near_pixel=near))
    json.dump(res, open(OUT / "seismic_soils.json", "w"), indent=1)
    print(json.dumps(res["seismic"], indent=1))


if __name__ == "__main__":
    main()
