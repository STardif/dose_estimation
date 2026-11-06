"""Electron-transport Monte Carlo, to cross-check the photoelectron-escape estimate.

A single-scattering Monte Carlo in the form described by D. C. Joy, *Monte
Carlo Modeling for Electron Microscopy and Microanalysis* (Oxford UP, 1995),
valid for electrons of ~0.5-50 keV:

* elastic scattering on the screened Rutherford cross-section (with the
  relativistic factor), screening parameter ``alpha = 3.4e-3 Z^0.67 / E``;
  the scattering atom is drawn in proportion to ``n_i sigma_i``;
* continuous energy loss along each step from the Joy-Luo modified Bethe
  stopping power, ``dE/ds = -78500 rho/E sum_i w_i Z_i/A_i ln(1.166 (E +
  k_i J_i)/J_i)`` keV/cm (Bragg additivity for compounds);
* tracking down to ``e_cut_kev`` (default 0.5 keV), the rest deposited
  locally.

It is checked against the Hunger-Kuechler fit of measured backscattering
coefficients and the ESTAR CSDA range of water (see the tests).

:func:`retention_mc` runs it for the photo- and Auger electrons released in a
material by a beam of given energy and profile, and returns the same
signal-weighted *mean* and *peak* retention as
:func:`dose_estimation.escape.component_retention`, optionally with a free
surface (bulk sample in reflection: electrons leaving the surface are lost)
and the Sauter angular distribution of K-shell photoelectrons, polarised along
the beam width (horizontal, as on a bending-magnet beamline).

This is not RADDOSE-3D, PENELOPE or Geant4: no inelastic secondaries (their
energy is deposited along the track) and no bremsstrahlung.  For the question
asked here, how much electron energy leaves a sub-micron footprint, it is an
independent check of the straight-track estimate, with physics that is
standard for electron-microprobe work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from . import atomic
from . import xray_data as xd
from .constants import AVOGADRO

__all__ = ["ElectronMedium", "simulate", "retention_mc", "backscatter_coefficient", "hunger_kuchler"]


class ElectronMedium:
    """Per-element constants of the screened-Rutherford / Joy-Luo model."""

    def __init__(self, mass_fractions: Dict[str, float], density: float):
        self.density = float(density)
        els = list(mass_fractions)
        self.symbols = els
        self.w = np.array([mass_fractions[e] for e in els])
        self.z = np.array([xd.atomic_number(e) for e in els], dtype=float)
        self.a = np.array([xd.atomic_weight(e) for e in els])
        self.n = AVOGADRO * self.density * self.w / self.a               # atoms/cm3
        self.j = (9.76 * self.z + 58.5 * self.z ** -0.19) * 1e-3          # keV
        self.k = 0.731 + 0.0688 * np.log10(self.z)

    def sigma(self, e_kev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Elastic cross-sections (cm2/atom), shape (n_el, n), and screening parameters."""
        e = np.maximum(e_kev, 1e-3)[None, :]
        alpha = 3.4e-3 * self.z[:, None] ** 0.67 / e
        sig = (5.21e-21 * self.z[:, None] ** 2 / e ** 2 * 4 * np.pi / (alpha * (1 + alpha))
               * ((e + 511.0) / (e + 1024.0)) ** 2)
        return sig, alpha

    def stopping(self, e_kev: np.ndarray) -> np.ndarray:
        """Stopping power, keV/cm (positive)."""
        e = np.maximum(e_kev, 1e-3)[None, :]
        arg = 1.166 * (e + self.k[:, None] * self.j[:, None]) / self.j[:, None]
        per = 78500.0 * self.density * (self.w * self.z / self.a)[:, None] / e * np.log(np.maximum(arg, 1.0001))
        return per.sum(axis=0)

    def csda_range_g_cm2(self, e_kev: float, n: int = 2000) -> float:
        """Continuous-slowing-down range from the Joy-Luo stopping power."""
        e = np.linspace(1e-3, e_kev, n)
        return float(np.trapezoid(1.0 / self.stopping(e), e) * self.density)


def _rotate(d: np.ndarray, cos_t: np.ndarray, psi: np.ndarray) -> np.ndarray:
    u, v, w = d[:, 0], d[:, 1], d[:, 2]
    sin_t = np.sqrt(np.clip(1 - cos_t ** 2, 0, None))
    cp, sp = np.cos(psi), np.sin(psi)
    s = np.sqrt(np.clip(1 - w ** 2, 0, None))
    out = np.empty_like(d)
    norm = s > 1e-6
    sn = np.where(norm, s, 1.0)
    out[:, 0] = np.where(norm, sin_t * (u * w * cp - v * sp) / sn + u * cos_t, sin_t * cp)
    out[:, 1] = np.where(norm, sin_t * (v * w * cp + u * sp) / sn + v * cos_t, sin_t * sp)
    out[:, 2] = np.where(norm, -sin_t * cp * s + w * cos_t, np.sign(w + 1e-300) * cos_t)
    out /= np.linalg.norm(out, axis=1)[:, None]
    return out


def simulate(medium: ElectronMedium, e0_kev, pos_um: np.ndarray, dir0: np.ndarray, rng,
             e_cut_kev: float = 0.5, surface: bool = False, on_deposit=None, max_steps: int = 200000):
    """Transport electrons; call ``on_deposit(index, positions_um, energies_kev)`` for every deposit.

    ``surface=True``: the medium fills z >= 0 and electrons crossing z < 0
    leave it (their remaining energy is returned as ``escaped_kev``).
    Returns ``(deposited_kev, escaped_kev)`` per electron.
    """
    n = len(pos_um)
    e = np.broadcast_to(np.asarray(e0_kev, dtype=float), (n,)).copy()
    pos = pos_um.astype(float).copy()
    d = dir0 / np.linalg.norm(dir0, axis=1)[:, None]
    alive = e > e_cut_kev
    deposited = np.zeros(n)
    escaped = np.zeros(n)
    # electrons born below the cut-off deposit on the spot
    if (~alive).any():
        idx = np.flatnonzero(~alive)
        deposited[idx] += e[idx]
        if on_deposit is not None:
            on_deposit(idx, pos[idx], e[idx])
    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        ek = e[idx]
        sig, alpha = medium.sigma(ek)
        inv_l = (medium.n[:, None] * sig).sum(axis=0)                     # 1/cm
        step_cm = -np.log(rng.random(idx.size)) / inv_l
        de = np.minimum(medium.stopping(ek) * step_cm, ek)
        new = pos[idx] + (step_cm * 1e4)[:, None] * d[idx]
        mid = 0.5 * (pos[idx] + new)
        if surface:
            out = new[:, 2] < 0
            # part of the step inside the medium
            frac = np.where(out, np.clip(pos[idx, 2] / np.maximum(pos[idx, 2] - new[:, 2], 1e-30), 0, 1), 1.0)
            de_in = de * frac
            mid = np.where(out[:, None], pos[idx] + 0.5 * (new - pos[idx]) * frac[:, None], mid)
        else:
            out = np.zeros(idx.size, dtype=bool)
            de_in = de
        deposited[idx] += de_in
        if on_deposit is not None:
            on_deposit(idx, mid, de_in)
        e[idx] = ek - de
        pos[idx] = new
        if surface and out.any():
            io = idx[out]
            escaped[io] += e[io] + (de - de_in)[out]
            e[io] = 0.0
            alive[io] = False
        # cut-off
        stop = alive[idx] & (e[idx] <= e_cut_kev)
        if stop.any():
            ist = idx[stop]
            deposited[ist] += e[ist]
            if on_deposit is not None:
                on_deposit(ist, pos[ist], e[ist].copy())
            e[ist] = 0.0
            alive[ist] = False
        # elastic scattering for the survivors
        cont = alive[idx]
        if cont.any():
            ic = idx[cont]
            p_el = (medium.n[:, None] * sig[:, cont])
            p_el = p_el / p_el.sum(axis=0)
            r = rng.random(ic.size)
            which = (r[None, :] > np.cumsum(p_el, axis=0)).sum(axis=0)
            which = np.minimum(which, len(medium.z) - 1)
            a = alpha[which, np.flatnonzero(cont)]
            rr = rng.random(ic.size)
            cos_t = 1 - 2 * a * rr / (1 + a - rr)
            psi = rng.uniform(0, 2 * np.pi, ic.size)
            d[ic] = _rotate(d[ic], cos_t, psi)
    return deposited, escaped


def hunger_kuchler(z: float, e_kev: float) -> float:
    """Backscattering coefficient at normal incidence, Hunger & Kuechler (1979) fit to measurements."""
    lz = math.log(z)
    m = 0.1382 - 0.9211 / math.sqrt(z)
    c = 0.1904 - 0.2235 * lz + 0.1292 * lz ** 2 - 0.01491 * lz ** 3
    return e_kev ** m * c


def backscatter_coefficient(medium: ElectronMedium, e0_kev: float, n: int = 4000, seed: int = 1,
                            e_cut_kev: float = 0.5) -> float:
    """Fraction of normally incident electrons leaving the surface (simulated)."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 3))
    pos[:, 2] = 1e-9
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    dep, esc = simulate(medium, e0_kev, pos, d, rng, e_cut_kev=e_cut_kev, surface=True)
    return float(np.mean(esc > 0))


def _sauter_directions(rng, n: int, e_kev: float, beam_dir: np.ndarray, pol: np.ndarray) -> np.ndarray:
    """K-shell photoelectron directions, Sauter: sin^2(t) cos^2(phi) / (1 - beta cos t)^4."""
    beta = math.sqrt(1 - 1 / (1 + e_kev / 511.0) ** 2)
    third = np.cross(beam_dir, pol)
    out = np.empty((0, 3))
    fmax = 1.0 / (1 - beta) ** 4
    while len(out) < n:
        m = 4 * (n - len(out)) + 64
        ct = rng.uniform(-1, 1, m)
        ph = rng.uniform(0, 2 * np.pi, m)
        st = np.sqrt(1 - ct ** 2)
        f = st ** 2 * np.cos(ph) ** 2 / (1 - beta * ct) ** 4
        keep = rng.random(m) * fmax < f
        v = (ct[keep, None] * beam_dir + (st[keep] * np.cos(ph[keep]))[:, None] * pol
             + (st[keep] * np.sin(ph[keep]))[:, None] * third)
        out = np.vstack([out, v])
    return out[:n]


def _isotropic(rng, n: int) -> np.ndarray:
    ct = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


@dataclass
class MCRetention:
    mean: float
    peak: float
    backscattered: float          # fraction of electron energy leaving through the surface
    n_electrons: int


def retention_mc(mass_fractions: Dict[str, float], density: float, photon_energy_ev: float,
                 width_um: float, height_um: float, profile: str = "gaussian",
                 grazing_angle_deg: Optional[float] = None, angular: str = "sauter",
                 n_electrons: int = 3000, seed: int = 7, e_cut_kev: float = 0.5,
                 attenuation_length_um: Optional[float] = None) -> MCRetention:
    """Fraction of the electron energy deposited in the beam footprint (Monte Carlo).

    Electrons are the photo- and Auger electrons of the elements of the
    material, in proportion to their share of ``mu_en`` at
    ``photon_energy_ev`` (same partition as :mod:`escape`).  Geometry:

    * ``grazing_angle_deg=None``: infinite medium, emission along the beam
      axis (lateral escape only, the assumption of :mod:`escape`);
    * ``grazing_angle_deg=40``: bulk sample with a free surface, beam at 40 deg
      from it, emission depth exponential along the beam with
      ``attenuation_length_um`` (computed from NIST if omitted).

    The beam travels along +y' in its own frame; width (horizontal, also the
    polarisation direction) along x, height in the plane of incidence.
    """
    rng = np.random.default_rng(seed)
    med = ElectronMedium(mass_fractions, density)
    e = float(photon_energy_ev)
    # electron sources
    srcs = []
    tot = 0.0
    for el, w in mass_fractions.items():
        share = w * float(xd.mu_en_rho(el, e)[0])
        eb, lab = atomic.binding_energy_below(el, e)
        omega = atomic.fluorescence_yield(el, lab) if lab else 0.0
        pe, au = e - eb, (1 - omega) * eb
        t = pe + au
        srcs.append(("photo", pe / 1e3, share * pe / t, lab))
        if au > 0:
            srcs.append(("auger", eb / 1e3, share * au / t, lab))
        tot += share
    weights = np.array([s[2] for s in srcs]) / tot
    counts = np.maximum(1, np.round(weights * n_electrons).astype(int))
    # geometry: slab normal z (into the sample), beam in the y-z plane
    if grazing_angle_deg is None:
        b = np.array([0.0, 0.0, 1.0])
        surface = False
    else:
        g = math.radians(grazing_angle_deg)
        b = np.array([0.0, math.cos(g), math.sin(g)])
        surface = True
    e1 = np.array([1.0, 0.0, 0.0])                     # width, polarisation
    e2 = np.cross(b, e1)                                # height, in the plane of incidence
    if profile == "gaussian":
        k = 1.0 / (2 * math.sqrt(2 * math.log(2)))
        s1, s2 = width_um * k, height_um * k
        p = lambda a, c: np.exp(-0.5 * (a / s1) ** 2 - 0.5 * (c / s2) ** 2) / (2 * math.pi * s1 * s2)  # noqa: E731
        sample = lambda m: (rng.normal(0, s1, m), rng.normal(0, s2, m))                               # noqa: E731
        p2 = 1 / (4 * math.pi * s1 * s2)
    else:
        p = lambda a, c: ((np.abs(a) <= width_um / 2) & (np.abs(c) <= height_um / 2)) / (width_um * height_um)  # noqa: E731
        sample = lambda m: (rng.uniform(-width_um / 2, width_um / 2, m), rng.uniform(-height_um / 2, height_um / 2, m))  # noqa: E731
        p2 = 1 / (width_um * height_um)
    p0 = p(np.zeros(1), np.zeros(1))[0]
    if surface and attenuation_length_um is None:
        from .materials import Material
        attenuation_length_um = float(Material(density=density, mass_fractions=mass_fractions)
                                      .attenuation_length_um(e)[0])
    acc_mean = 0.0
    acc_peak = 0.0
    acc_e = 0.0
    acc_bs = 0.0
    n_tot = 0
    for (kind, ee, wt, lab), m in zip(srcs, counts):
        if ee <= 0 or wt <= 0:
            continue
        a1, a2 = sample(m)
        if surface:
            # each ray enters the surface where z = 0, then is attenuated along its path
            s = a2 * (-e2[2]) / b[2] + rng.exponential(attenuation_length_um, m)
            origin = a1[:, None] * e1 + a2[:, None] * e2 + s[:, None] * b
        else:
            origin = a1[:, None] * e1 + a2[:, None] * e2
        if kind == "photo" and angular == "sauter":
            dirs = _sauter_directions(rng, m, ee, b, e1)
        else:
            dirs = _isotropic(rng, m)
        sm = np.zeros(m)
        sp = np.zeros(m)

        def on_dep(idx, xyz, de, origin=origin, sm=sm, sp=sp):
            t1 = xyz @ e1
            t2 = xyz @ e2
            sm[idx] += de * p(t1, t2)
            d1 = t1 - origin[idx] @ e1
            d2 = t2 - origin[idx] @ e2
            sp[idx] += de * p(-d1, -d2)

        dep, esc = simulate(med, ee, origin, dirs, rng, e_cut_kev=e_cut_kev, surface=surface, on_deposit=on_dep)
        # energy-weighted contributions of this source, normalised per electron energy
        acc_mean += wt * float(np.mean(sm) / (ee * p2))
        acc_peak += wt * float(np.mean(sp) / (ee * p0))
        acc_bs += wt * float(np.mean(esc) / ee)
        acc_e += wt
        n_tot += m
    return MCRetention(mean=acc_mean / acc_e, peak=acc_peak / acc_e, backscattered=acc_bs / acc_e,
                       n_electrons=n_tot)


def component_retention_mc(mass_fractions: Dict[str, float], density: float, energy_ev,
                           width_um: float, height_um: float, profile: str = "tophat",
                           grazing_angle_deg: Optional[float] = None, n_grid: int = 5,
                           n_electrons: int = 1500, seed: int = 7,
                           weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo retention over a spectrum, as ``(mean, peak)`` arrays over ``energy_ev``.

    The Monte Carlo runs on ``n_grid`` energies spanning the part of the
    spectrum that carries the dose (``weights``, e.g. the kerma contribution
    of each line); between them, the ratio Monte Carlo / analytic estimate is
    interpolated in log E and applied to the analytic retention, which keeps
    the jumps at absorption edges.
    """
    from . import escape
    e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
    an_m, an_p = escape.component_retention(mass_fractions, mass_fractions, density, e, width_um, height_um,
                                            profile)
    if e.size == 1:
        grid = e.copy()
    else:
        w = np.ones_like(e) if weights is None else np.clip(np.asarray(weights, dtype=float), 0, None)
        c = np.cumsum(w) / max(w.sum(), 1e-300)
        qs = np.linspace(0.05, 0.95, n_grid)
        grid = np.unique(np.interp(qs, c, e))
    rm, rp = [], []
    for k, eg in enumerate(grid):
        mc = retention_mc(mass_fractions, density, eg, width_um, height_um, profile, grazing_angle_deg,
                          "sauter", n_electrons=n_electrons, seed=seed + k)
        am, ap = escape.component_retention(mass_fractions, mass_fractions, density, [eg], width_um, height_um,
                                            profile)
        rm.append(mc.mean / max(am[0], 1e-12))
        rp.append(mc.peak / max(ap[0], 1e-12))
    if grid.size == 1:
        fm, fp = np.full_like(e, rm[0]), np.full_like(e, rp[0])
    else:
        lg = np.log(grid)
        fm = np.interp(np.log(e), lg, rm)
        fp = np.interp(np.log(e), lg, rp)
    return np.clip(an_m * fm, 0, 1), np.clip(an_p * fp, 0, 1)
