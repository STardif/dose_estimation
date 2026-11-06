"""Propagation of input uncertainties to the doses.

Two entry points:

* :func:`propagate` — generic Monte Carlo.  You give a function that builds
  and computes a :class:`DoseResult` from named inputs, and the
  distributions of those inputs (:class:`Uncertain`).  It returns the
  distribution of every component's dose (or dose rate) and the rank
  correlation of each output with each input.
* :func:`propagate_stack` — the same for an existing :class:`Stack` and
  :class:`Beam`, with the usual suspects perturbed: flux, beam width and
  height, layer thicknesses, porosity (the last component of each
  multi-component layer), material densities, photon energy and exposure.

Inputs are independent.  Multiplicative quantities (flux, sizes, thickness,
density, exposure) are log-normal with the given relative standard
deviation; porosity is normal (absolute standard deviation) and clipped to
the available volume.

A useful property of the model: the per-component dose depends on the
fluence (flux / area) and on the attenuation upstream, but not on the
component's own volume fraction or on the layer loading.  Porosity and
thickness uncertainties therefore matter only through attenuation and for
the mixed (electron-shared) dose; the fluence usually dominates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Union

import numpy as np
from scipy import stats

from .beam import Beam, Spectrum
from .materials import Material
from .stack import DoseResult, Layer, Stack

__all__ = ["Uncertain", "UncertaintyResult", "propagate", "propagate_stack", "dose_quantities"]


@dataclass
class Uncertain:
    """An uncertain input.

    ``rel`` (relative 1-sigma, log-normal around ``nominal``) or ``sd``
    (absolute 1-sigma, normal), optionally clipped to ``[low, high]``.
    """

    nominal: float
    rel: float = 0.0
    sd: float = 0.0
    low: float = -math.inf
    high: float = math.inf

    def sample(self, rng, n: int) -> np.ndarray:
        if self.rel and self.sd:
            raise ValueError("give rel or sd, not both")
        if self.rel:
            s = math.sqrt(math.log(1 + self.rel ** 2))
            x = self.nominal * np.exp(rng.normal(-0.5 * s ** 2, s, n))
        elif self.sd:
            x = rng.normal(self.nominal, self.sd, n)
        else:
            x = np.full(n, float(self.nominal))
        return np.clip(x, self.low, self.high)


def dose_quantities(res: DoseResult) -> Dict[str, float]:
    """Default outputs: dose (or dose rate without exposure) of every component and mixed layer, Gy."""
    out = {}
    for L in res.layers:
        for c in L.components:
            out[f"{L.name}/{c.material}"] = c.dose if c.dose is not None else c.dose_rate
        if len(L.components) > 1:
            out[f"{L.name}/(mixed)"] = L.mixed_dose if L.mixed_dose is not None else L.mixed_dose_rate
    return out


@dataclass
class UncertaintyResult:
    nominal: Dict[str, float]
    samples: Dict[str, np.ndarray]
    inputs: Dict[str, np.ndarray]
    unit_note: str = ""

    def _row(self, k):
        x = self.samples[k]
        p = np.percentile(x, [2.5, 16, 50, 84, 97.5])
        return dict(quantity=k, nominal=self.nominal[k], median=p[2], p16=p[1], p84=p[3], p2_5=p[0],
                    p97_5=p[4], rel_sigma=(p[3] - p[1]) / (2 * p[2]) if p[2] else float("nan"))

    def table(self):
        """Nominal, median, 68 % and 95 % intervals, relative 1-sigma (half the 16-84 % range / median)."""
        rows = [self._row(k) for k in self.samples]
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:  # pragma: no cover
            return rows

    def sensitivity(self):
        """Spearman rank correlation of each output (rows) with each input (columns)."""
        names = [n for n, v in self.inputs.items() if np.ptp(v) > 0]
        rows = []
        for k, y in self.samples.items():
            r = {"quantity": k}
            for n in names:
                r[n] = stats.spearmanr(self.inputs[n], y).statistic if np.ptp(y) > 0 else 0.0
            rows.append(r)
        try:
            import pandas as pd
            return pd.DataFrame(rows).set_index("quantity")
        except ImportError:  # pragma: no cover
            return rows

    def summary(self, scale: float = 1e-3, unit: str = "kGy") -> str:
        out = [f"{'quantity':40s} {'nominal':>11s} {'median':>11s} {'68 % interval':>25s} {'rel. 1-sigma':>12s}"]
        for k in self.samples:
            r = self._row(k)
            out.append(f"{k:40s} {r['nominal'] * scale:11.4g} {r['median'] * scale:11.4g} "
                       f"[{r['p16'] * scale:10.4g}, {r['p84'] * scale:10.4g}] {100 * r['rel_sigma']:11.1f} %")
        out.append(f"({unit}{self.unit_note})")
        return "\n".join(out)


def propagate(model: Callable[..., DoseResult], params: Mapping[str, Union[Uncertain, float]], n: int = 300,
              seed: int = 0, quantities: Callable[[DoseResult], Dict[str, float]] = dose_quantities
              ) -> UncertaintyResult:
    """Monte Carlo propagation through ``model(**inputs) -> DoseResult``.

    Example
    -------
    >>> def model(flux, width, height):
    ...     return cell.compute(Beam.monochromatic(27e3, flux, width, height), exposure_s=3600)
    >>> u = propagate(model, {"flux": Uncertain(2.44e10, rel=0.1),
    ...                       "width": Uncertain(630, rel=0.1), "height": Uncertain(110, rel=0.15)})
    >>> print(u.summary())
    """
    rng = np.random.default_rng(seed)
    specs = {k: (v if isinstance(v, Uncertain) else Uncertain(float(v))) for k, v in params.items()}
    nominal = quantities(model(**{k: v.nominal for k, v in specs.items()}))
    draws = {k: v.sample(rng, n) for k, v in specs.items()}
    samples = {k: np.empty(n) for k in nominal}
    for i in range(n):
        q = quantities(model(**{k: float(draws[k][i]) for k in specs}))
        for k in nominal:
            samples[k][i] = q[k]
    return UncertaintyResult(nominal=nominal, samples=samples, inputs=draws)


def propagate_stack(stack: Stack, beam: Beam, exposure_s: Optional[float] = None, *, flux_rel: float = 0.10,
                    size_rel: float = 0.10, thickness_rel: float = 0.0, porosity_sd: float = 0.0,
                    density_rel: float = 0.0, energy_rel: float = 0.0, exposure_rel: float = 0.0,
                    n: int = 300, seed: int = 0, quantities=dose_quantities, **compute_kw) -> UncertaintyResult:
    """Perturb an existing stack and beam.

    ``flux_rel``, ``size_rel`` (width and height independently),
    ``thickness_rel`` (each layer independently), ``density_rel`` (each
    material independently), ``energy_rel`` (all energies together) and
    ``exposure_rel`` are relative 1-sigma; ``porosity_sd`` is the absolute
    1-sigma of the volume fraction of the *last* component of every layer
    with two or more components (the filler), the others being rescaled to
    keep their sum.  ``compute_kw`` is passed to :meth:`Stack.compute`.
    """
    mats: List[Material] = []
    for L in stack.layers:
        for m, _ in L.components:
            if all(m is not x for x in mats):
                mats.append(m)
    params: Dict[str, Uncertain] = {
        "flux": Uncertain(1.0, rel=flux_rel), "width": Uncertain(1.0, rel=size_rel),
        "height": Uncertain(1.0, rel=size_rel), "energy": Uncertain(1.0, rel=energy_rel),
        "exposure": Uncertain(1.0, rel=exposure_rel)}
    for i, L in enumerate(stack.layers):
        if np.isfinite(L.thickness_um):
            params[f"thickness:{L.name}"] = Uncertain(1.0, rel=thickness_rel)
        if len(L.components) >= 2:
            tot = sum(v for _, v in L.components)
            params[f"porosity:{L.name}"] = Uncertain(L.components[-1][1], sd=porosity_sd, low=0.0, high=tot)
    for j, m in enumerate(mats):
        params[f"density:{m.name}#{j}"] = Uncertain(1.0, rel=density_rel)

    def model(**v):
        new_m = {id(m): (m if v[f"density:{m.name}#{j}"] == 1.0 else
                         m.with_density(m.density * v[f"density:{m.name}#{j}"]))
                 for j, m in enumerate(mats)}
        layers = []
        for L in stack.layers:
            comps = [(new_m[id(m)], f) for m, f in L.components]
            key = f"porosity:{L.name}"
            if key in v:
                tot = sum(f for _, f in comps)
                filler = min(max(v[key], 0.0), tot)
                rest_old = tot - comps[-1][1]
                scale = (tot - filler) / rest_old if rest_old > 0 else 0.0
                comps = [(m, f * scale) for m, f in comps[:-1]] + [(comps[-1][0], filler)]
                comps = [(m, f) for m, f in comps if f > 0] or comps
            t = L.thickness_um * v.get(f"thickness:{L.name}", 1.0)
            layers.append(Layer(L.name, components=comps, thickness_um=t))
        st = Stack(layers, incidence_angle_deg=stack.incidence_angle_deg)
        sp = beam.spectrum
        spec = Spectrum(sp.energy_ev * v["energy"], sp.photons_per_s * v["flux"], sp.label)
        b = Beam(spec, beam.width_um * v["width"], beam.height_um * v["height"], beam.profile)
        exp = None if exposure_s is None else exposure_s * v["exposure"]
        return st.compute(b, exposure_s=exp, **compute_kw)

    res = propagate(model, params, n=n, seed=seed, quantities=quantities)
    res.inputs = {k: a for k, a in res.inputs.items() if np.ptp(a) > 0}
    return res
