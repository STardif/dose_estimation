"""Steady-state temperature rise from the absorbed beam power (estimates).

SI-13 of the 2023 paper bounded the heating of the pouch cell with a 1D
conduction model; here the absorbed power comes from a :class:`DoseResult`
and three geometries are offered.  All are steady state (the upper bound for
a continuous exposure); the time to get there scales as ``a^2 / D_th`` (a
heated size, D_th the thermal diffusivity): microseconds to milliseconds for
micro-beams, seconds for mm beams on thin cells.

* ``through_thickness`` (SI-13): each layer heated uniformly in the beam,
  both faces held at ambient: ``dT = q_v t^2 / (8 k)``.  Relevant when the
  beam is much wider than the cell is thick and the cell is clamped between
  good conductors.
* ``lateral`` (thin stack, beam footprint >> thickness): the stack is a plate
  of in-plane conductance ``K = sum_j k_j t_j``; the absorbed power spreads
  radially to a heat sink at ``sink_radius_um`` (e.g. the edge of the cell or
  of a sample holder), or is lost from both faces with a coefficient
  ``h_w_m2k`` (convection + radiation, ~10 W/m^2/K in still air).  Disc
  source (top-hat) or Gaussian source, solved by numerical integration of
  the plate Green's function (``ln(R/r)`` with a sink, ``K0(m r)`` with
  losses, ``m = sqrt(2 h / K)``).
* ``bulk`` (semi-infinite last layer, micro-Laue in reflection): heat
  deposited along the beam path (transverse profile x attenuation in depth)
  in a half-space of conductivity k with an adiabatic surface (image
  source).  ``dT(r) = 1/(2 pi k) int q(r') / |r - r'| dV'`` evaluated by
  Monte Carlo along the beam axis; the maximum is reported.  Checks: a
  uniform disc on the surface gives ``P/(pi a k)``, a Gaussian surface source
  ``P/(2 sqrt(2 pi) k sigma)`` (Lax 1977).

Conductivities are inputs (W/m/K).  :data:`TYPICAL_CONDUCTIVITY` lists
room-temperature values for common materials; porous electrodes are far
below their bulk constituents (0.3-2 W/m/K is typical for coated electrodes
filled with electrolyte), so measure or look up the value for the actual
sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Union

import numpy as np
from scipy import integrate, special

from .constants import UM_PER_CM

__all__ = ["TYPICAL_CONDUCTIVITY", "ThermalEstimate", "temperature_rise", "plate_center_rise",
           "beam_path_rise", "bulk_max_rise", "through_thickness_rise", "halfspace_rise"]

#: Room-temperature thermal conductivities, W/m/K (bulk, dense materials).
TYPICAL_CONDUCTIVITY: Dict[str, float] = {
    "Cu": 401.0, "Al": 237.0, "Au": 318.0, "Ag": 429.0, "W": 173.0, "Ni": 91.0, "Fe": 80.0,
    "Si": 148.0, "Ge": 60.0, "diamond": 2000.0, "SiO2 (fused)": 1.4, "Al2O3": 30.0,
    "water": 0.6, "organic electrolyte": 0.2, "polymer (PP, PE, PVDF)": 0.2,
    "NMC (used in SI-13)": 10.0, "graphite (in-plane, HOPG)": 2000.0,
}

ConductivitySpec = Union[float, Mapping[str, float]]


@dataclass
class ThermalEstimate:
    """Temperature rises (K) and the inputs that produced them."""

    absorbed_power_w: float
    through_thickness_k: Dict[str, float] = field(default_factory=dict)
    lateral_k: Optional[float] = None
    near_field_k: Optional[float] = None
    bulk_k: Optional[float] = None
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def max_through_thickness_k(self) -> float:
        return max(self.through_thickness_k.values()) if self.through_thickness_k else 0.0

    @property
    def total_k(self) -> float:
        """Near-field rise (through-thickness, 3D or bulk) plus far-field spreading."""
        if self.bulk_k is not None:
            return self.bulk_k
        near = self.near_field_k if self.near_field_k is not None else self.max_through_thickness_k
        return near + (self.lateral_k or 0.0)

    def summary(self) -> str:
        out = [f"absorbed power {self.absorbed_power_w * 1e3:.4g} mW"]
        if self.through_thickness_k:
            worst = max(self.through_thickness_k, key=self.through_thickness_k.get)
            out.append(f"through thickness (SI-13): max {self.max_through_thickness_k:.3g} K in {worst}")
        if self.near_field_k is not None:
            out.append(f"near field (3D along the beam): {self.near_field_k:.3g} K")
        if self.lateral_k is not None:
            out.append(f"far field (spreading in the stack): {self.lateral_k:.3g} K")
        if self.bulk_k is not None:
            out.append(f"bulk (semi-infinite): {self.bulk_k:.3g} K")
        out.append(f"estimate: {self.total_k:.3g} K")
        return "; ".join(out)


def _k_of(layer, conductivity: ConductivitySpec) -> float:
    if isinstance(conductivity, (int, float)):
        return float(conductivity)
    if layer.name in conductivity:
        return float(conductivity[layer.name])
    # volume-weighted mean of the components found by material name (parallel bound)
    num, den = 0.0, 0.0
    for m, v in layer.components:
        if m.name not in conductivity:
            raise KeyError(f"no conductivity for layer {layer.name!r} or material {m.name!r}")
        num += v * float(conductivity[m.name])
        den += v
    return num / den


def through_thickness_rise(power_density_w_m3: float, thickness_um: float, k: float) -> float:
    """SI-13: uniform heating, both faces at ambient: ``q t^2 / (8 k)``."""
    t = thickness_um * 1e-6
    return power_density_w_m3 * t ** 2 / (8.0 * k)


def plate_center_rise(power_w: float, conductance_w_k: float, profile: str, width_um: float,
                      height_um: float, sink_radius_um: Optional[float] = None,
                      h_w_m2k: Optional[float] = None) -> float:
    """Centre temperature rise of a plate (in-plane conductance ``K = sum k t``, W/K).

    Elliptical sources are replaced by circular ones of the same area (top-hat)
    or the same ``sigma_x sigma_y`` (Gaussian).
    """
    if sink_radius_um is None and h_w_m2k is None:
        raise ValueError("give sink_radius_um (heat sink) and/or h_w_m2k (surface losses)")
    if profile == "tophat":
        a = math.sqrt(width_um * height_um / math.pi) * 1e-6

        def src(r):        # areal density, W/m2 per W
            return np.where(r <= a, 1.0 / (math.pi * a ** 2), 0.0)
        r_max = a
    else:
        k = 1.0 / (2 * math.sqrt(2 * math.log(2)))
        s = math.sqrt(width_um * height_um) * k * 1e-6

        def src(r):
            return np.exp(-0.5 * (r / s) ** 2) / (2 * math.pi * s ** 2)
        r_max = 12 * s
    if h_w_m2k is not None:
        m = math.sqrt(2.0 * h_w_m2k / conductance_w_k)
        R = sink_radius_um * 1e-6 if sink_radius_um is not None else None
        if R is None:
            def green(r):
                return special.k0(m * r) / (2 * math.pi * conductance_w_k)
        else:
            # fin with T = 0 at R (Dirichlet): K0 - I0 K0(mR)/I0(mR)
            def green(r):
                return (special.k0(m * r) - special.i0(m * r) * special.k0(m * R) / special.i0(m * R)) \
                    / (2 * math.pi * conductance_w_k)
    else:
        R = sink_radius_um * 1e-6

        def green(r):
            return np.log(R / r) / (2 * math.pi * conductance_w_k)
    if sink_radius_um is not None and sink_radius_um * 1e-6 <= r_max:
        raise ValueError("the heat sink must be outside the heated area")
    val, _ = integrate.quad(lambda r: src(r) * green(r) * 2 * math.pi * r, 0.0, r_max, limit=200,
                            points=[r_max / 3] if profile == "tophat" else None)
    return power_w * val


def beam_path_rise(result, k: float, n: int = 40000, seed: int = 3) -> float:
    """Maximum rise from the heat deposited along the beam path, in a half-space of conductivity ``k``.

    Heat sources are sampled in every layer in proportion to its absorbed
    power, along the beam (attenuation of each spectral line) and across it
    (beam profile); the surface z = 0 is adiabatic (image sources).  This is
    the near-field estimate for a beam smaller than the stack thickness, and
    the full estimate for a semi-infinite sample.
    """
    beam = result.beam
    rng = np.random.default_rng(seed)
    cosang = result._cache["cos"]
    sin_g = cosang                      # beam elevation above the surface = cos(incidence)
    cos_g = math.sqrt(max(0.0, 1 - sin_g ** 2))
    b = np.array([0.0, cos_g, sin_g])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(b, e1)
    powers = np.array([sum(c.absorbed_power_w for c in L.components) for L in result.layers])
    total = float(powers.sum())
    if total <= 0:
        return 0.0
    counts = rng.multinomial(n, powers / total)
    z_top = np.concatenate([[0.0], np.cumsum([L.thickness_um for L in result.layers])])
    pts = []
    for j, (L, m) in enumerate(zip(result.layers, counts)):
        if m == 0:
            continue
        spec = result.spectra_in[j]
        cache = result._cache["layers"][j]
        mu = cache["mu_star"] / 1e4                                             # 1/um
        w = spec.photons_per_s * spec.energy_ev * cache["dep_mixed"] / np.maximum(cache["mu_star"], 1e-300)
        path = L.thickness_um / cosang
        if np.isfinite(path):
            w = w * -np.expm1(-mu * path)
        w = w / w.sum()
        kk = rng.choice(len(w), size=m, p=w)
        u = rng.random(m)
        if np.isfinite(path):
            s = -np.log1p(-u * -np.expm1(-mu[kk] * path)) / mu[kk]
        else:
            s = -np.log1p(-u) / mu[kk]
        if beam.profile == "gaussian":
            s1, s2 = beam.sigma_um
            a1, a2 = rng.normal(0, s1, m), rng.normal(0, s2, m)
        else:
            a1 = rng.uniform(-beam.width_um / 2, beam.width_um / 2, m)
            a2 = rng.uniform(-beam.height_um / 2, beam.height_um / 2, m)
        # each ray enters layer j where z = z_top[j]
        s_tot = (z_top[j] - a2 * e2[2]) / max(b[2], 1e-9) + s
        pts.append(a1[:, None] * e1 + a2[:, None] * e2 + s_tot[:, None] * b)
    pts = np.vstack(pts)
    depth = np.percentile(np.linalg.norm(pts, axis=1), 90)
    axis = np.linspace(0.0, depth, 30)[:, None] * b
    floor = 0.02 * min(beam.width_um, beam.height_um)
    return float(np.max(halfspace_rise(pts, total, k, axis, floor_um=floor)))


def bulk_max_rise(result, k: float, n: int = 40000, seed: int = 3) -> float:
    """Maximum rise for a sample whose last layer is semi-infinite (see :func:`beam_path_rise`)."""
    if np.isfinite(result.layers[-1].thickness_um):
        raise ValueError("bulk model needs a semi-infinite last layer")
    return beam_path_rise(result, k, n=n, seed=seed)


def halfspace_rise(source_points_um: np.ndarray, power_w: float, k: float, eval_points_um: np.ndarray,
                   floor_um: float = 0.0) -> np.ndarray:
    """Steady rise (K) at ``eval_points_um`` from equal-weight heat sources in the half-space z >= 0.

    Adiabatic surface (image sources).  ``floor_um`` regularises 1/r for
    evaluation points that coincide with sources.
    """
    pts = np.asarray(source_points_um, dtype=float)
    img = pts * np.array([1.0, 1.0, -1.0])
    out = []
    for r0 in np.atleast_2d(eval_points_um):
        d1 = np.maximum(np.linalg.norm(pts - r0, axis=1), floor_um)
        d2 = np.maximum(np.linalg.norm(img - r0, axis=1), floor_um)
        out.append(power_w * np.mean(1.0 / d1 + 1.0 / d2) * 1e6 / (4 * math.pi * k))
    return np.array(out)


def temperature_rise(result, conductivity: ConductivitySpec, sink_radius_um: Optional[float] = None,
                     h_w_m2k: Optional[float] = None, bulk_conductivity: Optional[float] = None) -> ThermalEstimate:
    """Steady-state temperature-rise estimate for a :class:`DoseResult`.

    The estimate is a near-field term plus a far-field term:

    * near field: the SI-13 through-thickness model when the footprint is at
      least 3x wider than the stack is thick; otherwise (micro-beams, bulk
      samples) 3D conduction from the heat deposited along the beam path
      (:func:`beam_path_rise`), with the power-weighted conductivity of the
      layers;
    * far field (finite stacks only): radial spreading in the plate to a heat
      sink and/or surface losses (:func:`plate_center_rise`), with a source
      at least as wide as the stack is thick.

    Parameters
    ----------
    conductivity : float or mapping
        One value for every layer, or ``{layer name or material name: k}``
        (W/m/K); a layer without its own entry gets the volume-weighted mean
        of its components.
    sink_radius_um, h_w_m2k
        Boundary condition of the far-field model (at least one for finite
        stacks).
    bulk_conductivity : float, optional
        Conductivity used for a semi-infinite last layer (defaults to its
        entry in ``conductivity``).
    """
    est = ThermalEstimate(absorbed_power_w=sum(c.absorbed_power_w for c in result.components))
    beam = result.beam
    cosang = result._cache["cos"]
    layers = result.stack.layers
    ks = []
    for L in layers:
        if not np.isfinite(L.thickness_um) and bulk_conductivity is not None:
            ks.append(float(bulk_conductivity))
        else:
            ks.append(_k_of(L, conductivity))
    powers = np.array([sum(c.absorbed_power_w for c in LR.components) for LR in result.layers])
    # power-weighted harmonic mean: the heat sits mostly where it is deposited (conservative)
    k_eff = float(powers.sum() / np.sum(powers / np.array(ks))) if powers.sum() > 0 else ks[0]
    finite = all(np.isfinite(L.thickness_um) for L in layers)
    footprint = min(beam.width_um, beam.height_um / cosang)
    est.notes["footprint_um"] = footprint
    est.notes["k_eff_w_mk"] = k_eff
    if finite:
        t_stack = sum(L.thickness_um for L in layers)
        est.notes["stack_thickness_um"] = t_stack
        if footprint >= 3 * t_stack:
            # peak areal power density: footprint area (top-hat) or 2 pi sx sy (Gaussian), on the plate
            area_m2 = beam.nominal_area_cm2 * 1e-4 / cosang if beam.profile == "tophat" else \
                2 * math.pi * beam.sigma_um[0] * beam.sigma_um[1] * 1e-12 / cosang
            for LR, L, k in zip(result.layers, layers, ks):
                p = sum(c.absorbed_power_w for c in LR.components)
                vol = area_m2 * L.thickness_um * 1e-6
                est.through_thickness_k[L.name] = through_thickness_rise(p / vol, L.thickness_um, k)
            est.notes["near_field"] = "through thickness (SI-13)"
        else:
            est.near_field_k = beam_path_rise(result, k_eff)
            est.notes["near_field"] = "3D, along the beam path"
        if sink_radius_um is None and h_w_m2k is None:
            raise ValueError("finite stack: give sink_radius_um and/or h_w_m2k for the far field")
        K = sum(k * L.thickness_um * 1e-6 for L, k in zip(layers, ks))
        w, h = beam.width_um, beam.height_um / cosang
        prof = beam.profile
        if math.sqrt(w * h) < t_stack:            # the plate sees a source at least as wide as the stack
            w = h = math.sqrt(math.pi) * t_stack
            prof = "tophat"
        est.lateral_k = plate_center_rise(est.absorbed_power_w, K, prof, w, h, sink_radius_um, h_w_m2k)
        est.notes["in_plane_conductance_w_k"] = K
    else:
        est.bulk_k = beam_path_rise(result, k_eff)
        est.notes["near_field"] = "3D, semi-infinite"
    return est
