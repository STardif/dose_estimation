"""Photoelectron escape from small beams (an estimate).

The kerma ``Phi E mu_en/rho`` is the dose only if the secondary electrons stop
where they are created.  Photoelectrons from a 20 keV photon travel ~2 um in
NMC, ~4 um in graphite and ~8 um in an organic electrolyte; with a sub-micron
KB focus (micro-Laue) most of their energy is deposited *outside* the beam
footprint.  This module estimates the fraction that stays inside.

Model (deliberately simple, documented so it can be replaced):

* **Electron energies.**  An absorption on element *i* at photon energy E
  ejects a photoelectron of energy ``E - E_b`` (``E_b`` = highest edge of *i*
  below E, Elam data via :mod:`atomic`) and leaves a vacancy that relaxes by
  Auger emission with probability ``1 - omega`` (energy ~``E_b``) or by
  fluorescence (energy already excluded by ``mu_en``).  Each element contributes in proportion to
  its share of ``mu_en/rho``.
* **Range.**  Kanaya-Okayama range, ``R = 2.76e-6 A E^1.67 / Z^0.889`` g/cm^2
  (E in keV), combined over elements with Bragg additivity
  (``1/R = sum_i w_i / R_i``).  It reproduces the ESTAR CSDA range of water at
  20 keV within 4 %.
* **Track.**  Straight, isotropic, with the energy loss of a power-law range:
  ``E(s) = E0 (1 - s/R_eff)^(1/1.67)``, so deposition increases towards the
  end of the track.  Real tracks are tortuous, more so in heavy elements: the
  reach is the KO range times a *detour factor*.  By default it depends on
  the mass-weighted mean atomic number of the medium,
  ``detour = 1.72 Z^-0.357`` (clipped to [0.3, 1]), fitted to the electron
  Monte Carlo of :mod:`electron_mc` for C to Au at 12-30 keV (6 % rms, 15 %
  max).  The fit gives ~0.9 for organics, 0.65 for Si, 0.5 for Fe-Ge, 0.36
  for W-Au.  Emission anisotropy along the polarisation is ignored (the
  Monte Carlo shows it changes the retention by < 10 %).
* **Physical dose versus damage.**  The model estimates where the energy
  goes.  Micro-beam damage experiments on protein crystals at 18.5 keV
  found a smaller benefit than this physical escape: 3x less damage per
  dose from a 15.6 to a 0.84 um beam (Sanishvili et al. 2011), where both
  this model and the Monte Carlo predict ~7x less dose in the footprint.
  ``detour_factor=0.4`` comes close to the measured damage ratios (3.5x,
  and 2.3x for a 0.7 um line focus where Finfrock et al. 2010 measured
  2.0-2.5x); keep the kerma as the conservative bound when damage, not
  dose, is the question.
* **Retention.**  The lateral dose map is the beam profile convolved with the
  projected deposition kernel K.  For the *signal-weighted mean* dose the
  retained fraction is ``E_K[A(d)] / A(0)`` with A the autocorrelation of the
  profile; for the *peak* dose it is ``E_K[P(d)] / P(0)``.  Both are averaged
  over Monte Carlo samples of K (fixed seed, reproducible).

Only the lateral escape is treated: the beam is assumed to cross the whole
layer, and leakage across layer interfaces along the beam is neglected.
Electrons entering the footprint from material outside the beam are absent
by construction (nothing outside the beam is irradiated).  The estimate
agrees with the Monte Carlo of :mod:`electron_mc` within ~10 % (0.5 um beams,
10-25 keV, Si to NMC811); ``Stack.compute(electron_escape="mc")`` runs the
Monte Carlo itself.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from . import atomic
from . import xray_data as xd

__all__ = ["electron_range_um", "electron_range_g_cm2", "retention", "component_retention",
           "DEFAULT_DETOUR_FACTOR", "detour_factor_for"]

_KO_EXP = 1.67
_N_SAMPLES = 40000


def electron_range_g_cm2(mass_fractions: Dict[str, float], energy_kev) -> np.ndarray:
    """Kanaya-Okayama range in g/cm^2 (Bragg-additive over elements)."""
    e = np.atleast_1d(np.asarray(energy_kev, dtype=float))
    inv = 0.0
    for el, w in mass_fractions.items():
        z = xd.atomic_number(el)
        a = xd.atomic_weight(el)
        inv = inv + w * z ** 0.889 / a
    return 2.76e-6 * np.power(np.maximum(e, 1e-6), _KO_EXP) / inv


def electron_range_um(mass_fractions: Dict[str, float], density: float, energy_kev) -> np.ndarray:
    """Kanaya-Okayama range in um for a medium of the given density (g/cm^3)."""
    return electron_range_g_cm2(mass_fractions, energy_kev) / density * 1e4


@lru_cache(maxsize=4)
def _unit_kernel(n: int = _N_SAMPLES, seed: int = 20230519) -> Tuple[np.ndarray, np.ndarray]:
    """Projected (x, y) displacements of energy deposition for R = 1."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    s = 1.0 - (1.0 - u) ** _KO_EXP              # inverse CDF of the deposition along the track
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    rho = s * np.sqrt(1.0 - cos_t ** 2)
    return rho * np.cos(phi), rho * np.sin(phi)


#: ``None`` = Z-dependent detour factor (see :func:`detour_factor_for`).
DEFAULT_DETOUR_FACTOR = None


def detour_factor_for(mass_fractions: Dict[str, float]) -> float:
    """Detour factor fitted to the electron Monte Carlo: ``1.72 Zbar^-0.357`` in [0.3, 1]."""
    zbar = sum(w * xd.atomic_number(el) for el, w in mass_fractions.items())
    return float(np.clip(1.72 * max(zbar, 1.0) ** -0.357, 0.3, 1.0))


def retention(range_um, width_um: float, height_um: float, profile: str = "tophat") -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of the electron energy retained in the beam footprint.

    Returns ``(mean, peak)`` for the signal-weighted mean dose and for the
    dose at the beam centre, for each range in ``range_um``.
    """
    r = np.atleast_1d(np.asarray(range_um, dtype=float))
    ux, uy = _unit_kernel()
    mean = np.empty_like(r)
    peak = np.empty_like(r)
    if profile == "gaussian":
        k = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sx, sy = width_um * k, height_um * k
    for i, ri in enumerate(r):
        dx = ri * ux
        dy = ri * uy
        if profile == "tophat":
            ax = np.clip(1.0 - np.abs(dx) / width_um, 0.0, None)
            ay = np.clip(1.0 - np.abs(dy) / height_um, 0.0, None)
            mean[i] = np.mean(ax * ay)
            peak[i] = np.mean((np.abs(dx) < width_um / 2) & (np.abs(dy) < height_um / 2))
        else:
            mean[i] = np.mean(np.exp(-dx ** 2 / (4 * sx ** 2) - dy ** 2 / (4 * sy ** 2)))
            peak[i] = np.mean(np.exp(-dx ** 2 / (2 * sx ** 2) - dy ** 2 / (2 * sy ** 2)))
    return mean, peak


def component_retention(mass_fractions: Dict[str, float], medium_fractions: Dict[str, float],
                        medium_density: float, energy_ev, width_um: float, height_um: float,
                        profile: str = "tophat",
                        detour_factor: Optional[float] = DEFAULT_DETOUR_FACTOR) -> Tuple[np.ndarray, np.ndarray]:
    """Energy-resolved retention for the electrons released in a material.

    Parameters
    ----------
    mass_fractions : dict
        Composition of the material that absorbs (sets the electron energies).
    medium_fractions, medium_density
        Composition and density of the medium the electrons travel in (the
        material itself, or the homogenised layer).
    energy_ev : array
        Photon energies.
    detour_factor : float or None
        Effective reach / Kanaya-Okayama range; None = Z-dependent default
        for the medium (see the module docstring).

    Returns ``(mean, peak)`` arrays over ``energy_ev``.
    """
    e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
    # electron energies and weights per element
    shares = []
    men_tot = np.zeros_like(e)
    for el, w in mass_fractions.items():
        men = w * xd.mu_en_rho(el, e)
        men_tot += men
        shares.append((el, men))
    # ranges on a log grid, then interpolate retention in log(range)
    e_list, w_list, idx_list = [], [], []
    for el, men in shares:
        for k, ek in enumerate(e):
            eb, lab = atomic.binding_energy_below(el, ek)
            omega = atomic.fluorescence_yield(el, lab) if lab else 0.0
            pe, au = ek - eb, (1.0 - omega) * eb
            tot = pe + au
            frac = men[k] / men_tot[k] if men_tot[k] > 0 else 0.0
            e_list.append(pe / 1e3); w_list.append(frac * pe / tot); idx_list.append(k)
            if au > 0:
                e_list.append(eb / 1e3); w_list.append(frac * au / tot); idx_list.append(k)
    e_el = np.array(e_list)
    if detour_factor is None:
        detour_factor = detour_factor_for(medium_fractions)
    if not 0 < detour_factor <= 1.5:
        raise ValueError("detour_factor should be in (0, 1.5]")
    ranges = detour_factor * electron_range_um(medium_fractions, medium_density, e_el)
    grid = np.geomspace(max(ranges.min(), 1e-4), max(ranges.max(), 1e-4) * 1.0001, 64)
    gm, gp = retention(grid, width_um, height_um, profile)
    lr = np.log(np.clip(ranges, grid[0], grid[-1]))
    rm = np.interp(lr, np.log(grid), gm)
    rp = np.interp(lr, np.log(grid), gp)
    mean = np.zeros_like(e)
    peak = np.zeros_like(e)
    np.add.at(mean, idx_list, np.array(w_list) * rm)
    np.add.at(peak, idx_list, np.array(w_list) * rp)
    return mean, peak
