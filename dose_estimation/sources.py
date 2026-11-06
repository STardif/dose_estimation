"""Incident spectra from beamline models.

White-beam (micro-Laue) doses need the absolute spectrum at the sample,
photons/s/eV.  For BM32 (ESRF) it is computed by ``xrf_beamline`` of pyXRT
(https://github.com/STardif/pyXRT): bending-magnet source, Ir mirrors, Be
windows, slits and the incident air path, with the vertical acceptance traced
through the optics.  This module wraps it into a :class:`~dose_estimation.Spectrum`.

pyXRT is a set of modules rather than an installed package, and it needs
``xraylib``.  Point to it with ``pyxrt_path=...`` or the ``PYXRT_PATH``
environment variable, or put its folder on ``sys.path`` yourself.

The pyXRT spectrum already includes the incident air path to the sample
(3.3 m by default): do not add it again as a layer of the stack.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Optional

import numpy as np

from .beam import Beam, Spectrum

__all__ = ["spectrum_from_pyxrt", "bm32_white_beam", "bm32_white_beam_from_h5",
           "DEFAULT_WHITE_GRID_KEV", "import_xrf_beamline"]

#: Default energy grid for white-beam spectra, keV (the BM32 band is ~5-28 keV).
DEFAULT_WHITE_GRID_KEV = np.round(np.arange(3.0, 40.0 + 1e-9, 0.025), 6)


def import_xrf_beamline(pyxrt_path: Optional[str] = None):
    """Import pyXRT's ``xrf_beamline`` module.

    Looks in ``pyxrt_path``, then ``$PYXRT_PATH``, then the current
    ``sys.path``.
    """
    for p in (pyxrt_path, os.environ.get("PYXRT_PATH")):
        if p:
            p = os.path.abspath(os.path.expanduser(p))
            if not os.path.isfile(os.path.join(p, "xrf_beamline.py")):
                raise ImportError(f"no xrf_beamline.py in {p}")
            if p not in sys.path:
                sys.path.insert(0, p)
            break
    try:
        return importlib.import_module("xrf_beamline")
    except ImportError as exc:
        raise ImportError(
            "Could not import pyXRT's xrf_beamline (and its dependency xraylib). "
            "Clone https://github.com/STardif/pyXRT, `pip install xraylib`, and pass "
            "pyxrt_path=... or set PYXRT_PATH.") from exc


def spectrum_from_pyxrt(beamline, energies_kev=None, horizontal_acceptance_urad: Optional[float] = None,
                        label: str = "") -> Spectrum:
    """Absolute spectrum at the sample from a pyXRT ``Beamline``.

    Uses ``Beamline.photon_flux`` (photons/s/eV at the sample, accepted
    horizontal fan included).  ``energies_kev`` defaults to
    :data:`DEFAULT_WHITE_GRID_KEV`; it must be fine enough to resolve the
    mirror cut-off and the Ir L edges (25 eV is plenty).
    """
    e_kev = np.asarray(DEFAULT_WHITE_GRID_KEV if energies_kev is None else energies_kev, dtype=float)
    if np.any(e_kev < 1.0):
        raise ValueError("white-beam grid must start at >= 1 keV (NIST table range)")
    dens = np.asarray(beamline.photon_flux(e_kev, horizontal_acceptance_urad), dtype=float)
    dens = np.nan_to_num(np.clip(dens, 0.0, None))
    return Spectrum.from_density(e_kev * 1e3, dens, label or "pyXRT white beam")


def bm32_white_beam(energies_kev=None, pyxrt_path: Optional[str] = None,
                    horizontal_acceptance_urad: Optional[float] = None, **beamline_kwargs) -> Spectrum:
    """BM32 white-beam spectrum at the sample (photons/s per line).

    ``beamline_kwargs`` go to ``xrf_beamline.BM32_WHITE_BEAM`` (slit gaps
    ``vg1 hg1 ... vg3 hg3`` in mm, mirror angles and radii, ``current_a``,
    ``air_path_m``, ...).  The default slit settings are those of the 2026
    micro-Laue maps.
    """
    xb = import_xrf_beamline(pyxrt_path)
    bl = xb.BM32_WHITE_BEAM(**beamline_kwargs)
    return spectrum_from_pyxrt(bl, energies_kev, horizontal_acceptance_urad, label="BM32 white beam")


def bm32_white_beam_from_h5(h5_path: str, scan: Optional[str] = None, energies_kev=None,
                            pyxrt_path: Optional[str] = None,
                            horizontal_acceptance_urad: Optional[float] = None, **overrides) -> Spectrum:
    """BM32 white-beam spectrum with the slits, mirrors and ring current of a BLISS scan."""
    xb = import_xrf_beamline(pyxrt_path)
    bl = xb.bm32_from_h5(h5_path, scan, **overrides)
    return spectrum_from_pyxrt(bl, energies_kev, horizontal_acceptance_urad,
                               label=f"BM32 white beam ({os.path.basename(str(h5_path))})")
