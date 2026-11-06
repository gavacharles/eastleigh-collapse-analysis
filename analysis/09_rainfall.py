"""Step 9 - antecedent rainfall at the site (tests the 'wet foundation soils' trigger)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from credit import footer

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "gis" / "raw" / "rain_openmeteo.json"))["daily"]
t = [datetime.fromisoformat(x) for x in d["time"]]
p = np.array([v or 0.0 for v in d["precipitation_sum"]])
cum = np.cumsum(p[::-1])[::-1]          # rain from each day up to the collapse

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(t, p, width=0.8, color="#1f6aa5", label="daily rainfall [mm]")
ax2 = ax.twinx()
ax2.plot(t, cum, color="#c77700", lw=1.6, label="rain from that day to 23 Oct [mm]")
ax.axvline(t[-1], color="#c62f2f", lw=1.5)
ax.text(t[-1], ax.get_ylim()[1] * 0.92, " collapse\n 23 Oct", color="#c62f2f", fontsize=8, va="top")
for days in (7, 30):
    ax.text(0.01, 0.92 - 0.09 * (days == 30), f"last {days} days: {p[-days:].sum():.0f} mm",
            transform=ax.transAxes, fontsize=9, family="monospace")
ax.set(ylabel="daily rainfall [mm]", title="Antecedent rainfall at the site (Jul-Oct 2026)")
ax2.set_ylabel("cumulative to collapse [mm]")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
footer(fig, "Open-Meteo daily precipitation (weather-model/reanalysis blend, ~11 km grid; not a rain gauge)")
fig.savefig(HERE / "gis" / "fig_rainfall.png", dpi=150)
json.dump(dict(rain_7d=float(p[-7:].sum()), rain_30d=float(p[-30:].sum()), rain_92d=float(p.sum()),
               max_day=float(p.max()), max_day_date=d["time"][int(p.argmax())]),
          open(HERE / "gis" / "rain_stats.json", "w"), indent=1)
print(open(HERE / "gis" / "rain_stats.json").read())
