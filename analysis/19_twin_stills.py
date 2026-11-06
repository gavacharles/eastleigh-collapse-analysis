"""
Step 19 - Digital-twin visualisation: four 3-D stills of the FE twin (scenario S6).

  a  intact: L0 column demand/capacity under gravity
  b  corner pair lost: redistributed dynamic demand, deformed shape (x40)
  c  cascade step 1: neighbouring columns exceed capacity
  d  full ground-storey loss: the superstructure has no support
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from credit import SRC_TWIN, stamp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "fe"
pv.OFF_SCREEN = True
SCEN = "S6 as-built f'c 15, two corner-end columns lost"


def main():
    fe = json.load(open(OUT / "fe_anim.json"))
    geo, sc = fe["geo"], fe["anim"][SCEN]
    xs = np.array(geo["xs"]) - geo["xs"][-1] / 2
    ys = np.array(geo["ys"]) - geo["ys"][-1] / 2
    zs = np.array(geo["zs"])
    nx, ny = len(xs), len(ys)
    nodes = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    du = np.array(sc["u_after"]) - np.array(sc["u_intact"])
    steps = sc["steps"]
    parse = lambda d: {tuple(int(v) for v in k.strip("()").split(",")): val for k, val in d.items()}
    removed_after = [set(map(tuple, s["removed"])) | set(map(tuple, s.get("new_failures", []))) for s in steps]
    panels = [("a  Intact: L0 demand / capacity", parse(steps[0]["dcr"]), set(), 0.0),
              ("b  Corner pair lost: dynamic redistribution (deformation x40)", parse(steps[1]["dcr"]), set(map(tuple, steps[1]["removed"])), 40.0),
              ("c  Cascade step 1: neighbours exceed capacity", parse(steps[2]["dcr"]) if len(steps) > 2 else parse(steps[1]["dcr"]),
               set(map(tuple, steps[1]["removed"])) | set(map(tuple, steps[1].get("new_failures", []))), 40.0),
              ("d  Full ground-storey loss: no support left", {}, {(i, j) for i in range(nx) for j in range(ny)}, 0.0)]
    cmap = plt.get_cmap("RdYlGn_r")
    tiles = []
    for title, dcr, gone, amp in panels:
        p = pv.Plotter(off_screen=True, window_size=(800, 900))
        p.set_background("#15191c")
        p.add_mesh(pv.Plane(center=(0, 0, -0.05), i_size=50, j_size=50), color="#2a3137")
        P = nodes + du * amp
        segs = []
        for a, b in geo["beams"]:
            segs.append((P[a], P[b]))
        for n, storey, ij, size in geo["cols"]:
            if storey > 1:
                segs.append((P[n[0]], P[n[1]]))
        p.add_mesh(pv.line_segments_from_points(np.array(segs).reshape(-1, 3)), color="#c9d1d6", line_width=1.4)
        for j in range(ny):
            for i in range(nx):
                top = P[(1 * ny + j) * nx + i]
                if (i, j) in gone:
                    p.add_mesh(pv.Sphere(radius=0.8, center=(xs[i], ys[j], 0.6)), color="#ff3b30")
                    continue
                d = dcr.get((i, j), 0)
                p.add_mesh(pv.Cylinder(center=(xs[i], ys[j], top[2] / 2), direction=(0, 0, 1), radius=0.45,
                                       height=top[2]), color=cmap(min(d, 1.2) / 1.2)[:3])
                if d > 0:
                    p.add_point_labels([(xs[i], ys[j], top[2] + 0.6)], [f"{d:.2f}"], font_size=11, text_color="white",
                                       shape=None, show_points=False, always_visible=True)
        p.camera_position = [(-52, -62, 42), (0, 0, 18), (0, 0, 1)]
        im = np.ascontiguousarray(p.screenshot(return_img=True))
        p.close()
        cv2.rectangle(im, (0, 0), (im.shape[1], 40), (10, 12, 14), -1)
        cv2.putText(im, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 200, 60), 2, cv2.LINE_AA)
        tiles.append(im)
    bar = np.full((60, 1600, 3), 21, np.uint8)
    for k in range(400):
        c = (np.array(cmap(k / 400)[:3]) * 255).astype(int).tolist()
        cv2.line(bar, (40 + k, 15), (40 + k, 35), c, 1)
    cv2.putText(bar, "0", (36, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(bar, "1.2", (430, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(bar, "L0 column demand / capacity (labels: DCR; red spheres: failed or removed columns)",
                (470, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:]), bar])
    grid = stamp(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), SRC_TWIN, "scenario S6 (as-built f'c 15 MPa, two corner-end columns lost), DAF 2",
                 scale=0.55)
    cv2.imwrite(str(OUT / "fig_twin_3d_stages.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("saved", OUT / "fig_twin_3d_stages.jpg")


if __name__ == "__main__":
    main()
