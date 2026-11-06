"""Compare RADDOSE-3D photoelectron escape with dose_estimation (see make_inputs.py)."""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from make_inputs import CASES, ENERGIES_KEV  # noqa: E402
from dose_estimation import Material, Layer, Stack, Beam  # noqa: E402


def summary(prefix):
    with open(os.path.join(HERE, prefix + "Summary.csv")) as fh:
        rows = list(csv.reader(fh))
    head = [h.strip() for h in rows[0]]
    vals = [float(x) for x in rows[1]]
    return dict(zip(head, vals))


def main():
    print(f"{'case':16s}{'RD3D DWD ratio':>16s}{'RD3D max ratio':>16s}{'analytic mean':>15s}{'MC mean':>9s}{'analytic peak':>15s}")
    for name, c in CASES.items():
        m = Material(c["formula"], c["density"], name)
        st = Stack([Layer.solid(name, m, 8.0)])
        for e in ENERGIES_KEV:
            try:
                on = summary(f"{name}_{e:g}keV_petrue_")
                off = summary(f"{name}_{e:g}keV_pefalse_")
            except FileNotFoundError:
                print(f"{name} {e:g} keV: RADDOSE-3D output missing")
                continue
            beam = Beam.monochromatic(e * 1e3, 1e10, 0.5, 0.5, profile="gaussian")
            an = st.compute(beam, electron_escape="analytic").components[0]
            mc = st.compute(beam, electron_escape="mc").components[0]
            print(f"{name + ' ' + format(e, 'g') + ' keV':16s}{on['Average DWD'] / off['Average DWD']:16.3f}"
                  f"{on['Max Dose'] / off['Max Dose']:16.3f}{an.retention_mean:15.3f}{mc.retention_mean:9.3f}"
                  f"{an.retention_peak:15.3f}")


if __name__ == "__main__":
    main()
