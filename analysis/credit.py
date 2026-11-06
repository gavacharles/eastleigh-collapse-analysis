"""Shared attribution: every visualisation shows its data sources and the credit line."""
import cv2
import numpy as np

CREDIT = "Credit: Gava Charles"

SRC_VIDEO = "Eyewitness video VID-20261023-WA0009 (WhatsApp, 23 Oct 2026)"
SRC_CV = "optical-flow tracking (this analysis)"
SRC_MODEL = "crush-up ODE + PINN fit (this analysis)"
SRC_TWIN = "parametric RC frame model, assumed sections/loads (this analysis)"
SRC_OSM = "OpenStreetMap contributors (ODbL), via Overpass API"
SRC_ESRI = "Esri World Imagery (Esri, Maxar, Earthstar Geographics)"
SRC_ESRI_GREY = "Esri World Light Gray Canvas"
SRC_DEM = "AWS Terrain Tiles (Mapzen/Tilezen; SRTM-derived in Kenya)"
SRC_PRESS = "press reports: Kenyans.co.ke, Africanews/AP (23 Oct 2026)"


def line(*sources):
    return "Data: " + "; ".join(sources) + "   |   " + CREDIT


def footer(fig, *sources, fontsize=7.5):
    """Bottom-left sources + credit on a matplotlib figure (call before savefig)."""
    fig.text(0.005, 0.003, line(*sources), fontsize=fontsize, color="0.25", ha="left", va="bottom",
             wrap=True)


def stamp(img, *sources, scale=0.42):
    """Append a dark caption strip with sources + credit to a BGR image."""
    w = img.shape[1]
    text = line(*sources)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # wrap to width
    words, lines, cur = text.split(" "), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if cv2.getTextSize(trial, font, scale, 1)[0][0] > w - 12:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    lines.append(cur)
    lh = int(22 * scale / 0.42)
    strip = np.full((lh * len(lines) + 8, w, 3), 20, np.uint8)
    for k, ln in enumerate(lines):
        cv2.putText(strip, ln, (6, lh * (k + 1)), font, scale, (225, 225, 225), 1, cv2.LINE_AA)
    return np.vstack([img, strip])

SITE_LABEL = "LR No. 36/II/40, 9th Street, Eastleigh, Kamukunji Sub-County, Nairobi"
