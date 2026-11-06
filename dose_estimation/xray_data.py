"""Elemental X-ray interaction data.

Two coefficients matter for a dose calculation and they must not be confused:

* the **mass attenuation coefficient** ``mu/rho`` (cm^2/g) gives the
  probability that a photon *interacts* (photo-absorption, coherent and
  incoherent scattering).  It governs how the beam is attenuated on its way
  through the sample (Beer-Lambert law);
* the **mass energy-absorption coefficient** ``mu_en/rho`` (cm^2/g) gives the
  fraction of the photon energy that is *deposited locally* by the secondary
  electrons.  It excludes the energy carried away by scattered photons
  (Rayleigh, Compton) and by fluorescence photons.  The dose (kerma, under
  charged-particle equilibrium) is ``D = Phi * E * mu_en/rho`` for a fluence
  ``Phi``.

Both are read from the NIST tables of Hubbell & Seltzer (NISTIR 5632, 1995,
updated 2004), 1 keV - 20 MeV, Z = 1-92, shipped in ``data/NIST_xray_data``.
Interpolation is done piecewise between absorption edges with a monotone cubic
(PCHIP) in log-log space, which is what NIST recommends for these tables;
the edges are kept as true discontinuities.

An optional ``xraylib`` backend can be selected with :func:`set_backend`; it is
useful below 1 keV, where the NIST tables stop.
"""

from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = [
    "ElementData",
    "element",
    "atomic_number",
    "atomic_weight",
    "symbol_of",
    "mu_rho",
    "mu_en_rho",
    "edges",
    "set_backend",
    "get_backend",
    "NIST_E_MIN_EV",
    "NIST_E_MAX_EV",
]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_NIST_DIR = os.path.join(_DATA_DIR, "NIST_xray_data")
_WEIGHTS_FILE = os.path.join(_DATA_DIR, "molw.dat")

NIST_E_MIN_EV = 1.0e3
NIST_E_MAX_EV = 2.0e7

_BACKEND = "nist"


def set_backend(name: str) -> None:
    """Select the cross-section source: ``"nist"`` (default, bundled) or ``"xraylib"``.

    ``xraylib`` must be importable.  Its ``CS_Total`` and ``CS_Energy`` are used
    for ``mu/rho`` and ``mu_en/rho``; edge energies still come from the NIST
    files.
    """
    global _BACKEND
    name = name.lower()
    if name not in ("nist", "xraylib"):
        raise ValueError("backend must be 'nist' or 'xraylib'")
    if name == "xraylib":
        import xraylib  # noqa: F401  (fail early if missing)
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


# -----------------------------------------------------------------------------
# Atomic weights
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _weights() -> Dict[str, Tuple[int, float, str]]:
    table: Dict[str, Tuple[int, float, str]] = {}
    with open(_WEIGHTS_FILE, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("atomic_number") or not line.strip():
                continue
            z, sym, name, w = line.strip().split(",")[:4]
            table[sym] = (int(z), float(w), name)
    return table


def atomic_number(symbol: str) -> int:
    try:
        return _weights()[symbol][0]
    except KeyError:
        raise ValueError(f"Unknown element symbol {symbol!r}") from None


def atomic_weight(symbol: str) -> float:
    """Standard atomic weight, g/mol (IUPAC 2021)."""
    try:
        return _weights()[symbol][1]
    except KeyError:
        raise ValueError(f"Unknown element symbol {symbol!r}") from None


@lru_cache(maxsize=None)
def symbol_of(z: int) -> str:
    for sym, (zz, _, _) in _weights().items():
        if zz == z:
            return sym
    raise ValueError(f"No element with Z = {z}")


# -----------------------------------------------------------------------------
# NIST tables
# -----------------------------------------------------------------------------

@dataclass
class ElementData:
    """Tabulated NIST data for one element.

    Attributes
    ----------
    symbol, z, weight
    energy_ev : array
        Tabulated energies, strictly increasing within each edge segment.
    mu_rho, mu_en_rho : array
        cm^2/g.
    edges : dict
        Absorption-edge label -> energy (eV), for the edges listed by NIST
        (those above 1 keV).
    """

    symbol: str
    z: int
    weight: float
    energy_ev: np.ndarray
    mu_rho_tab: np.ndarray
    mu_en_rho_tab: np.ndarray
    edges: Dict[str, float] = field(default_factory=dict)
    _segments: List[Tuple[float, float, PchipInterpolator, PchipInterpolator]] = field(
        default_factory=list, repr=False)

    def __post_init__(self):
        self._build()

    def _build(self) -> None:
        e = self.energy_ev
        # An edge appears as two rows a few eV apart (below / above); split the
        # table there so that the interpolation never smooths across an edge.
        cut = [0]
        for i in range(1, len(e)):
            if e[i] - e[i - 1] <= 2.0 + 1e-9 or e[i] / e[i - 1] < 1.0 + 1e-4:
                cut.append(i)
        cut.append(len(e))
        segs = []
        for a, b in zip(cut[:-1], cut[1:]):
            le = np.log(e[a:b])
            lm = np.log(self.mu_rho_tab[a:b])
            lme = np.log(self.mu_en_rho_tab[a:b])
            if b - a == 1:
                # single point segment: constant extrapolation within it
                le = np.array([le[0], le[0] + 1e-9])
                lm = np.repeat(lm, 2)
                lme = np.repeat(lme, 2)
            segs.append((e[a], e[b - 1], PchipInterpolator(le, lm, extrapolate=True),
                         PchipInterpolator(le, lme, extrapolate=True)))
        self._segments = segs

    def _eval(self, energy_ev, which: int) -> np.ndarray:
        e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
        if np.any(e < NIST_E_MIN_EV * (1 - 1e-9)) or np.any(e > NIST_E_MAX_EV * (1 + 1e-9)):
            raise ValueError(
                f"Energy outside the NIST table range (1 keV - 20 MeV) for {self.symbol}: "
                f"{e.min():.4g}-{e.max():.4g} eV.  Use set_backend('xraylib') below 1 keV.")
        out = np.empty_like(e)
        lows = np.array([s[0] for s in self._segments])
        # segment index: last segment whose lower bound is <= e
        idx = np.clip(np.searchsorted(lows, e * (1 + 1e-10), side="right") - 1, 0, len(self._segments) - 1)
        le = np.log(e)
        for k in np.unique(idx):
            m = idx == k
            out[m] = np.exp(self._segments[k][2 + which](le[m]))
        return out

    def mu_rho(self, energy_ev) -> np.ndarray:
        """Mass attenuation coefficient, cm^2/g (total, with coherent scattering)."""
        return self._eval(energy_ev, 0)

    def mu_en_rho(self, energy_ev) -> np.ndarray:
        """Mass energy-absorption coefficient, cm^2/g."""
        return self._eval(energy_ev, 1)


def _nist_file(z: int) -> str:
    hits = glob.glob(os.path.join(_NIST_DIR, f"nist_xray_data_Z{z:02d}_*.dat"))
    if not hits:
        raise ValueError(f"No NIST data file for Z = {z} (only Z = 1-92 are tabulated)")
    return hits[0]


@lru_cache(maxsize=None)
def element(symbol: str) -> ElementData:
    """Load (and cache) the NIST data of an element."""
    z = atomic_number(symbol)
    e, mu, men, labels = [], [], [], []
    with open(_nist_file(z), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            lab = ""
            if len(parts) == 4:
                lab, parts = parts[0], parts[1:]
            if len(parts) != 3:
                continue
            e.append(round(float(parts[0]) * 1e6, 4))
            mu.append(float(parts[1]))
            men.append(float(parts[2]))
            labels.append(lab)
    e = np.array(e)
    # The scraped files place each edge 1 eV above its tabulated energy; when a
    # regular grid point falls inside that 1 eV (Mo K, Bi M1, Fr M5) the rows
    # are out of order.  A stable sort restores a valid table.
    order = np.argsort(e, kind="stable")
    e = e[order]
    mu = np.array(mu)[order]
    men = np.array(men)[order]
    labels = [labels[i] for i in order]
    # remove exact duplicates (keep the first)
    keep = np.concatenate([[True], np.diff(e) > 0])
    edges = {lab: float(en) for lab, en in zip(labels, e) if lab}
    return ElementData(symbol=symbol, z=z, weight=atomic_weight(symbol), energy_ev=e[keep],
                       mu_rho_tab=mu[keep], mu_en_rho_tab=men[keep], edges=edges)


def edges(symbol: str) -> Dict[str, float]:
    """Absorption edges (eV) of an element, as listed in the NIST table (> 1 keV)."""
    return dict(element(symbol).edges)


# -----------------------------------------------------------------------------
# Public coefficient functions (backend aware)
# -----------------------------------------------------------------------------

def mu_rho(symbol: str, energy_ev) -> np.ndarray:
    """Mass attenuation coefficient of an element, cm^2/g."""
    if _BACKEND == "xraylib":
        import xraylib
        z = atomic_number(symbol)
        e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
        return np.array([xraylib.CS_Total(z, ek / 1e3) for ek in e])
    return element(symbol).mu_rho(energy_ev)


def mu_en_rho(symbol: str, energy_ev) -> np.ndarray:
    """Mass energy-absorption coefficient of an element, cm^2/g."""
    if _BACKEND == "xraylib":
        import xraylib
        z = atomic_number(symbol)
        e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
        return np.array([xraylib.CS_Energy(z, ek / 1e3) for ek in e])
    return element(symbol).mu_en_rho(energy_ev)


# -----------------------------------------------------------------------------
# Atomic relaxation (approximate), used by the electron-escape estimate
# -----------------------------------------------------------------------------

def fluorescence_yield(z: int, shell: str) -> float:
    """Approximate fluorescence yield.

    K shell: Bambynek (1984) fit, ``(w/(1-w))^(1/4) = 0.015 + 0.0327 Z - 0.64e-6 Z^3``
    (better than 5 % above Z = 15).  L shells: ``Z^4 / (Z^4 + 8.9e7)``, a coarse
    Burhop-type average of the L subshells.  M and higher: 0.
    """
    if shell.startswith("K"):
        x = (0.015 + 0.0327 * z - 0.64e-6 * z ** 3) ** 4
        return x / (1.0 + x)
    if shell.startswith("L"):
        return z ** 4 / (z ** 4 + 8.9e7)
    return 0.0


def binding_energy_below(symbol: str, energy_ev: float) -> Tuple[float, str]:
    """Highest absorption edge of ``symbol`` below ``energy_ev``: (energy, label).

    Returns ``(0, "")`` when no tabulated edge (> 1 keV) lies below, i.e. the
    photoelectron takes essentially all of the photon energy.
    """
    best = (0.0, "")
    for lab, en in element(symbol).edges.items():
        if en < energy_ev and en > best[0]:
            best = (en, lab)
    return best


def _selftest() -> None:  # pragma: no cover - used during development
    for s, e, ref_mu, ref_men in [("O", 10e3, 5.952, 5.565), ("Cu", 20e3, 33.79, 27.88)]:
        assert math.isclose(float(mu_rho(s, e)[0]), ref_mu, rel_tol=1e-6)
        assert math.isclose(float(mu_en_rho(s, e)[0]), ref_men, rel_tol=1e-6)
