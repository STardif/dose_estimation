"""White-beam integration with pyXRT (skipped unless xraylib and pyXRT are available).

Set PYXRT_PATH to a clone of https://github.com/STardif/pyXRT.
"""

import os

import numpy as np
import pytest

pytest.importorskip("xraylib")
if not os.environ.get("PYXRT_PATH"):
    pytest.skip("PYXRT_PATH not set", allow_module_level=True)

from dose_estimation import Beam, Layer, Material, Stack            # noqa: E402
from dose_estimation.sources import bm32_white_beam, import_xrf_beamline  # noqa: E402


def test_bm32_spectrum_matches_pyxrt_total_flux():
    xb = import_xrf_beamline()
    bl = xb.BM32_WHITE_BEAM()
    e = np.arange(3.0, 40.0, 0.025)
    spec = bm32_white_beam(energies_kev=e)
    assert spec.total_flux == pytest.approx(bl.total_flux(e), rel=1e-9)
    lo, hi = spec.band(0.01)
    assert 4e3 < lo < 10e3 and 20e3 < hi < 35e3


def test_white_beam_dose_runs():
    spec = bm32_white_beam(energies_kev=np.arange(3.0, 40.0, 0.05))
    beam = Beam(spec, 0.5, 0.5, profile="gaussian")
    st = Stack([Layer.solid("Si", Material("Si", 2.33), np.inf)], grazing_angle_deg=40)
    r = st.compute(beam, exposure_s=1.0, electron_escape=True)
    c = r.components[0]
    assert 0 < c.dose_rate < c.kerma_rate
