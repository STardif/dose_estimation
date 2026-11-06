"""Reabsorption of fluorescence photons inside the beam footprint.

``mu_en`` assumes that every fluorescence photon leaves the volume of
interest.  For a large beam, part of them is reabsorbed in the irradiated
volume (Ni Ka has a 38 um attenuation length in dense NMC811, comparable to
a 110 um beam height), so the true dose lies between ``mu_en`` and
``mu``.  This module computes the reabsorbed part with a Monte Carlo of the
fluorescence photons through the whole stack.

Model
-----
1. **Sources.**  For each layer j, component c and element i, the
   photo-absorption rate is ``sum_E N_j(E) v_c rho_c w_i tau_i(E) / mu*_j(E)
   (1 - exp(-mu*_j L_j))``.  The photo-absorption coefficient is taken from
   NIST through ``tau_i = (mu_en/rho)_i / (1 - X_i(E)/E)``, the definition of
   ``mu_en`` (Hubbell & Seltzer), with ``X_i`` the fluorescence energy per
   absorption.  Shell fractions come from jump ratios, vacancies are moved
   by L-shell Coster-Kronig transitions, and each level emits its lines with
   the Elam fluorescence yields and intensities (:mod:`atomic`).
2. **Emission points** follow the primary attenuation along the beam (energy
   sampled from the spectrum, depth from a truncated exponential), with the
   transverse position drawn from the beam profile; emission is isotropic.
3. **Transport.**  Each photon travels through the laterally infinite stack
   with the total attenuation coefficient of each layer at the line energy,
   until it is absorbed or leaves the stack.  At the absorption point it
   deposits ``E_f mu_en/mu`` (secondary fluorescence and scattering of the
   fluorescence photon are dropped), shared between the components of that
   layer in proportion to ``v_c rho_c (mu_en/rho)_c``.
4. **Footprint.**  A deposit counts with the beam-profile density at its
   transverse position, i.e. the same signal weighting as the primary dose.
   The result is a ratio ``r_c`` = reabsorbed fluorescence energy / primary
   energy, both signal-weighted, per component, so that the dose becomes
   ``D (1 + r_c)``.  The same ratio is applied to the peak dose (an upper
   bound: the fluorescence dose is smoother than the primary one).

Lines below 1 keV (L lines of 3d metals, K lines of light elements) are
absorbed within a fraction of a micron; they are deposited where they are
emitted, in the emitting component.

Geometry: the beam is tilted in the plane containing the *height* direction
(``Beam.height_um`` lies in the plane of incidence, as for BM32 micro-Laue
in reflection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import atomic
from . import xray_data as xd
from .constants import J_PER_EV, UM_PER_CM

__all__ = ["FluorescenceReport", "reabsorption"]

_E_LOCAL_EV = 1000.0      # lines below this are deposited locally


@dataclass
class FluorescenceReport:
    """Diagnostics of the fluorescence calculation (powers in W)."""

    emitted_w: float = 0.0
    local_w: float = 0.0                 # soft lines deposited where emitted
    reabsorbed_in_stack_w: float = 0.0   # absorbed anywhere in the stack
    escaped_w: float = 0.0               # left the stack
    skipped_w: float = 0.0               # sources too weak to be simulated
    n_sources: int = 0
    ratios: Dict[Tuple[int, int], float] = field(default_factory=dict)       # (layer, component) index
    mixed_ratios: Dict[int, float] = field(default_factory=dict)             # layer index

    def summary(self) -> str:
        tot = max(self.emitted_w, 1e-300)
        return (f"fluorescence emitted {self.emitted_w:.3e} W in {self.n_sources} sources: "
                f"{100 * self.reabsorbed_in_stack_w / tot:.1f} % absorbed in the stack, "
                f"{100 * self.escaped_w / tot:.1f} % escaped, {100 * self.local_w / tot:.1f} % soft (local)")


def _profile(beam):
    if beam.profile == "tophat":
        w, h = beam.width_um, beam.height_um

        def p(x1, x2):
            return ((np.abs(x1) <= w / 2) & (np.abs(x2) <= h / 2)) / (w * h)

        def sample(rng, n):
            return rng.uniform(-w / 2, w / 2, n), rng.uniform(-h / 2, h / 2, n)

        p2 = 1.0 / (w * h)
    else:
        s1, s2 = beam.sigma_um

        def p(x1, x2):
            return np.exp(-0.5 * (x1 / s1) ** 2 - 0.5 * (x2 / s2) ** 2) / (2 * math.pi * s1 * s2)

        def sample(rng, n):
            return rng.normal(0, s1, n), rng.normal(0, s2, n)

        p2 = 1.0 / (4 * math.pi * s1 * s2)
    return p, sample, p2


def _trunc_exp(rng, rate, length, n):
    """Sample s in [0, length) with density ~ exp(-rate s) (length may be inf)."""
    u = rng.random(n)
    rate = np.maximum(rate, 1e-300)
    if np.isinf(length).all() if np.ndim(length) else math.isinf(length):
        return -np.log1p(-u) / rate
    return -np.log1p(-u * (-np.expm1(-rate * length))) / rate


def reabsorption(stack, beam, spectra_in, primary_power, cosang: float, n_photons: int = 20000,
                 seed: int = 12345, min_power_fraction: float = 1e-4) -> FluorescenceReport:
    """Fluorescence reabsorbed in the footprint, as ratios to the primary signal-weighted energy.

    Parameters
    ----------
    stack : Stack
    beam : Beam
    spectra_in : list of Spectrum
        Spectrum entering each layer (from the primary calculation).
    primary_power : dict
        ``{(layer index, component index): absorbed primary power (W)}`` (mu_en based).
    cosang : float
        cos of the incidence angle.
    """
    rng = np.random.default_rng(seed)
    p_fun, p_sample, p2 = _profile(beam)
    layers = stack.layers
    n_l = len(layers)
    z_edges = np.concatenate([[0.0], np.cumsum([L.thickness_um for L in layers])])     # um, normal
    sin_a = math.sqrt(max(0.0, 1.0 - cosang ** 2))
    # transverse unit vectors in slab coordinates: e1 = x (width), e2 = (0, cos a, -sin a) (height)
    e2 = np.array([0.0, cosang, -sin_a])

    rep = FluorescenceReport()
    deposit: Dict[Tuple[int, int], float] = {}           # (layer, component) -> sum eps * p
    local: Dict[Tuple[int, int], float] = {}             # soft lines: power, deposited as primary-like

    # --- enumerate sources ----------------------------------------------------------------
    sources = []    # (layer j, component c, E_f, rate photons/s, weights over E, mu*_j(E), L_cm)
    for j, L in enumerate(layers):
        spec = spectra_in[j]
        e = spec.energy_ev
        nj = spec.photons_per_s
        if nj.sum() <= 0:
            continue
        mu_star = L.mu(e)
        l_cm = L.thickness_um / cosang / UM_PER_CM
        absorbed = np.ones_like(e) if not np.isfinite(l_cm) else -np.expm1(-mu_star * l_cm)
        for c, (m, v) in enumerate(L.components):
            for el, w in m.mass_fractions.items():
                if xd.atomic_number(el) < 3:
                    continue
                men = xd.mu_en_rho(el, e)
                # lines per absorption, energy-dependent (edges inside the spectrum)
                # group the spectrum by the set of open shells to keep this cheap
                xi = np.array([atomic.fluorescence_energy_per_absorption(el, ek) for ek in e])
                if not np.any(xi > 0):
                    continue
                tau = men / np.clip(1.0 - xi / e, 1e-6, None)                      # cm2/g
                n_abs = nj * v * m.density * w * tau / mu_star * absorbed          # photo-absorptions/s per line E
                # lines emitted, by spectral line (edges change the set of lines)
                line_rate: Dict[float, np.ndarray] = {}
                for k, ek in enumerate(e):
                    if n_abs[k] <= 0:
                        continue
                    for ef, y in atomic.fluorescence_lines(el, ek):
                        key = round(ef, 1)
                        if key not in line_rate:
                            line_rate[key] = np.zeros_like(e)
                        line_rate[key][k] += n_abs[k] * y
                for ef, wts in line_rate.items():
                    rate = float(wts.sum())
                    power = rate * ef * J_PER_EV
                    rep.emitted_w += power
                    if ef < _E_LOCAL_EV:
                        rep.local_w += power
                        local[(j, c)] = local.get((j, c), 0.0) + power
                        continue
                    sources.append((j, c, ef, rate, wts / rate, mu_star, l_cm, power))
    if not sources:
        return rep
    tot_emit = sum(s[7] for s in sources)

    # --- transport -----------------------------------------------------------------------
    for (j, c, ef, rate, wts, mu_star_j, l_cm, power) in sources:
        if power < min_power_fraction * tot_emit:
            rep.skipped_w += power
            continue
        rep.n_sources += 1
        n = int(n_photons)
        mu_f = np.array([float(L.mu(ef)[0]) for L in layers]) / UM_PER_CM            # 1/um
        # emission point
        kidx = rng.choice(len(wts), size=n, p=wts)
        s_cm = _trunc_exp(rng, mu_star_j[kidx], l_cm, n)
        z0 = z_edges[j] + s_cm * UM_PER_CM * cosang
        if np.isfinite(z_edges[j + 1]):
            z0 = np.minimum(z0, z_edges[j + 1] * (1 - 1e-12))
        x1, x2 = p_sample(rng, n)
        # direction
        ct = rng.uniform(-1.0, 1.0, n)
        st = np.sqrt(1.0 - ct ** 2)
        ph = rng.uniform(0.0, 2 * math.pi, n)
        d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
        # march through layers with the optical depth
        tau_rem = -np.log1p(-rng.random(n))
        layer = np.full(n, j)
        z = z0.copy()
        dist = np.zeros(n)
        alive = np.ones(n, dtype=bool)
        absorbed_in = np.full(n, -1)
        for _ in range(4 * n_l + 4):
            if not alive.any():
                break
            idx = np.flatnonzero(alive)
            lk = layer[idx]
            mz = d[idx, 2]
            up = mz < 0
            bound = np.where(up, z_edges[lk], z_edges[lk + 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                to_b = np.where(np.abs(mz) > 1e-12, (bound - z[idx]) / mz, np.inf)
            to_b = np.where(np.isfinite(to_b), np.maximum(to_b, 0.0), np.inf)
            mu_k = mu_f[lk]
            to_abs = tau_rem[idx] / mu_k
            hit = to_abs <= to_b
            # absorbed here
            ih = idx[hit]
            dist[ih] += to_abs[hit]
            absorbed_in[ih] = lk[hit]
            alive[ih] = False
            # cross the boundary
            im = idx[~hit]
            step = to_b[~hit]
            dist[im] += step
            z[im] = z[im] + step * mz[~hit]
            tau_rem[im] -= step * mu_k[~hit]
            nxt = lk[~hit] + np.where(up[~hit], -1, 1)
            out = (nxt < 0) | (nxt >= n_l)
            alive[im[out]] = False
            layer[im[~out]] = nxt[~out]
        w_ph = rate / n                                                               # photons/s each
        got = absorbed_in >= 0
        rep.reabsorbed_in_stack_w += got.sum() * w_ph * ef * J_PER_EV
        rep.escaped_w += (~got).sum() * w_ph * ef * J_PER_EV
        if not got.any():
            continue
        dd = dist[got]
        t1 = x1[got] + dd * d[got, 0]
        t2 = x2[got] + dd * (d[got] @ e2)
        weight = p_fun(t1, t2) * w_ph * ef * J_PER_EV                               # W / um2
        lay = absorbed_in[got]
        for k in np.unique(lay):
            sel = lay == k
            Lk = layers[k]
            mu_tot = float(Lk.mu(ef)[0])
            wsum = float(weight[sel].sum())
            for ci, (mc, vc) in enumerate(Lk.components):
                share = vc * mc.density * float(mc.mu_en_rho(ef)[0]) / mu_tot
                deposit[(k, ci)] = deposit.get((k, ci), 0.0) + wsum * share

    # --- ratios ----------------------------------------------------------------------------
    for k, L in enumerate(layers):
        num_mix, den_mix = 0.0, 0.0
        for ci, (m, v) in enumerate(L.components):
            prim = primary_power.get((k, ci), 0.0)
            num = deposit.get((k, ci), 0.0) + local.get((k, ci), 0.0) * p2
            den = prim * p2
            rep.ratios[(k, ci)] = num / den if den > 0 else 0.0
            num_mix += num
            den_mix += den
        rep.mixed_ratios[k] = num_mix / den_mix if den_mix > 0 else 0.0
    return rep
