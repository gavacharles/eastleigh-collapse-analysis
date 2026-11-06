"""Step 4 - annotated key frames (visual evidence panel)."""
from pathlib import Path

import cv2
import numpy as np

from credit import SRC_VIDEO, stamp

VIDEO = Path(__file__).resolve().parent.parent / "VID-20261023-WA0009.mp4"
OUT = Path(__file__).resolve().parent / "out"

KEY = [  # frame, crop (x0,y0,x1,y1), caption
    (0, (40, 300, 400, 620), "0.00s pre-existing rubble, ladders, open L0"),
    (24, (40, 300, 400, 620), "0.80s left end of L1 slab dropping"),
    (40, (0, 450, 480, 800), "1.33s L1 slab descends into L0"),
    (52, (0, 450, 480, 800), "1.74s lateral air/dust jets at L0"),
    (104, (0, 0, 480, 864), "3.47s upper block intact, vertical"),
    (160, (0, 0, 480, 864), "5.34s remnant in dust (core?)"),
]


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    tiles = []
    for i, (x0, y0, x1, y1), cap_txt in KEY:
        im = frames[i][y0:y1, x0:x1]
        im = cv2.resize(im, (420, int(420 * im.shape[0] / im.shape[1])))
        im = cv2.resize(im, (420, 560)) if im.shape[0] > 560 else im
        pad = np.full((560, 420, 3), 255, np.uint8)
        pad[:im.shape[0]] = im
        cv2.rectangle(pad, (0, 0), (420, 30), (0, 0, 0), -1)
        cv2.putText(pad, cap_txt, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(pad)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    grid = stamp(grid, SRC_VIDEO, "annotations: this analysis", scale=0.5)
    cv2.imwrite(str(OUT / "fig_evidence_frames.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
