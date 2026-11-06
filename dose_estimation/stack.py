"""Layers, stacks and the dose calculation.

Model
-----
A :class:`Stack` is an ordered list of :class:`Layer` (upstream first).  A layer
is a homogenised mixture of components, each a :class:`~dose_estimation.Material`
with a volume fraction ``v_c`` (a porous electrode: solid 1 - eps, electrolyte
eps; any remainder is void).  The beam crosses the stack at
``incidence_angle_deg`` from the normal, so each path length is ``t / cos``.

For every photon energy E of the beam spectrum, in layer j:

* attenuation: ``mu*(E) = sum_c v_c rho_c (mu/rho)_c``; the incident flux on
  layer j+1 is ``N_j(E) exp(-mu* L_j)`` (beam hardening is automatic);
* dose rate in component c, depth-averaged over the layer (Gy/s)::

      D_c = sum_E  f N_j(E)/A * E * (mu_en/rho)_c * (1 - exp(-x)) / x,   x = mu* L_j

  with ``A`` the nominal beam area and ``f`` the profile factor (1 for a
  top-hat; see :class:`~dose_estimation.Beam`).  This is SI-6 of Jousseaume
  et al. (2023) with ``mu_en`` in place of ``mu`` for the deposited energy;
  ``deposition="attenuation"`` restores the 2023 behaviour;
* the *mixed* dose of the layer, ``sum_c P_c / sum_c m_c``, is the limit where
  secondary electrons are shared between the components (electron range much
  larger than the particle size).  The true dose of a minor phase lies between
  its own per-component value and this mixed value.

Optionally, :mod:`~dose_estimation.escape` corrects for photoelectrons leaving
the beam footprint (micro-beams).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .beam import Beam, Spectrum
from .constants import CM2_PER_UM2, J_PER_EV, UM_PER_CM
from .materials import Material
from . import escape as _escape
from . import fluorescence as _fluo

__all__ = ["Layer", "Stack", "DoseResult", "ComponentDose", "LayerDose"]

Component = Tuple[Material, float]


class Layer:
    """A homogenised layer of one or more components.

    Parameters
    ----------
    name : str
    components : list of (Material, volume fraction) or a single Material
        Volume fractions must sum to at most 1; the remainder is void.
    thickness_um : float
        Thickness along the layer normal, um.  ``numpy.inf`` for a bulk
        (semi-infinite) sample, e.g. in reflection geometry.

    Convenience constructors: :meth:`Layer.porous`, :meth:`Layer.solid`.

    The v0 signature ``Layer(name, solidMaterial, liquidMaterial, thickness,
    porosity)`` (thickness in **mm**) is still accepted.
    """

    def __init__(self, name: str, *args, components=None, thickness_um: Optional[float] = None,
                 solidMaterial: Optional[Material] = None, liquidMaterial: Optional[Material] = None,
                 thickness: Optional[float] = None, porosity: Optional[float] = None):
        legacy = (solidMaterial is not None or liquidMaterial is not None or thickness is not None
                  or porosity is not None)
        if args:
            # positional: (components, thickness_um) or v0 (solid, liquid, thickness_mm, porosity)
            v0_positional = (len(args) >= 3
                             or (len(args) == 2 and (args[1] is None or isinstance(args[1], Material)))
                             or (len(args) == 1 and legacy))
            if v0_positional:
                legacy = True
                names = ["solidMaterial", "liquidMaterial", "thickness", "porosity"]
                vals = dict(zip(names, args))
                solidMaterial = vals.get("solidMaterial", solidMaterial)
                liquidMaterial = vals.get("liquidMaterial", liquidMaterial)
                thickness = vals.get("thickness", thickness)
                porosity = vals.get("porosity", porosity)
            else:
                components = args[0]
                if len(args) > 1:
                    thickness_um = args[1]
        self.name = str(name)
        if legacy:
            if thickness is None:
                raise TypeError("v0 Layer: thickness (mm) is required")
            eps = 0.0 if porosity is None else float(porosity)
            if not 0.0 <= eps <= 1.0:
                raise ValueError("porosity must be within [0, 1]")
            comps: List[Component] = []
            if eps < 1.0:
                if solidMaterial is None:
                    raise ValueError(f"layer {name!r}: porosity < 1 needs a solidMaterial")
                comps.append((solidMaterial, 1.0 - eps))
            if eps > 0.0:
                if liquidMaterial is None:
                    raise ValueError(f"layer {name!r}: porosity > 0 needs a liquidMaterial "
                                     "(use Layer.porous(..., filler=None) for empty pores)")
                comps.append((liquidMaterial, eps))
            elif liquidMaterial is not None:
                warnings.warn(f"layer {name!r}: porosity is 0, liquidMaterial ignored", stacklevel=2)
            components = comps
            thickness_um = float(thickness) * 1e3
            self.porosity = eps
            self.solidMaterial = solidMaterial if eps < 1.0 else None
            self.liquidMaterial = liquidMaterial if eps > 0.0 else None
        if components is None or thickness_um is None:
            raise TypeError("Layer needs components and thickness_um")
        if isinstance(components, Material):
            components = [(components, 1.0)]
        comps = [(m, float(v)) for m, v in components]
        if not comps:
            raise ValueError("a layer needs at least one component")
        for m, v in comps:
            if not isinstance(m, Material):
                raise TypeError("components must be (Material, volume_fraction) pairs")
            if v < 0:
                raise ValueError("volume fractions must be >= 0")
        tot = sum(v for _, v in comps)
        if tot <= 0:
            raise ValueError(f"layer {name!r}: volume fractions sum to zero")
        if tot > 1.0 + 1e-9:
            raise ValueError(f"layer {name!r}: volume fractions sum to {tot:.4g} > 1")
        if not (thickness_um > 0):
            raise ValueError("thickness must be positive")
        self.components: List[Component] = comps
        self.thickness_um = float(thickness_um)
        if not legacy:
            # v0-style attributes, for code that inspects them
            self.porosity = 1.0 - comps[0][1]
            self.solidMaterial = comps[0][0]
            self.liquidMaterial = comps[1][0] if len(comps) > 1 else None

    @classmethod
    def porous(cls, name: str, solid: Material, porosity: float, filler: Optional[Material],
               thickness_um: float) -> "Layer":
        """Solid with a porosity filled by ``filler`` (``None`` = empty pores)."""
        if not 0.0 <= porosity <= 1.0:
            raise ValueError("porosity must be within [0, 1]")
        comps = []
        if porosity < 1:
            comps.append((solid, 1.0 - porosity))
        if porosity > 0 and filler is not None:
            comps.append((filler, porosity))
        return cls(name, components=comps, thickness_um=thickness_um)

    @classmethod
    def solid(cls, name: str, material: Material, thickness_um: float) -> "Layer":
        return cls(name, components=[(material, 1.0)], thickness_um=thickness_um)

    @property
    def thickness(self) -> float:
        """v0 name: thickness in mm."""
        return self.thickness_um / 1e3

    @property
    def density(self) -> float:
        """Mean density of the layer, g/cm^3 (voids count as empty)."""
        return sum(v * m.density for m, v in self.components)

    @property
    def mass_fractions(self) -> Dict[str, float]:
        """Elemental mass fractions of the homogenised layer."""
        rho = self.density
        w: Dict[str, float] = {}
        for m, v in self.components:
            for el, f in m.mass_fractions.items():
                w[el] = w.get(el, 0.0) + f * v * m.density / rho
        return w

    def mu(self, energy_ev) -> np.ndarray:
        """Effective linear attenuation coefficient, 1/cm."""
        return sum(v * m.mu(energy_ev) for m, v in self.components)

    def __repr__(self) -> str:
        comp = ", ".join(f"{m.name} {v:.3g}" for m, v in self.components)
        return f"Layer({self.name!r}, {self.thickness_um:g} um: {comp})"


@dataclass
class ComponentDose:
    """Dose to one component of one layer.  Rates in Gy/s, doses in Gy."""

    layer: str
    material: str
    volume_fraction: float
    density: float
    mass_g: Optional[float]
    interaction_fraction: float
    interaction_fraction_layer: float
    energy_fraction: float
    absorbed_power_w: float
    kerma_rate: float
    kerma_rate_max: float
    retention_mean: float
    retention_peak: float
    dose_rate: float
    dose_rate_max: float
    dose: Optional[float] = None
    dose_max: Optional[float] = None
    fluorescence_ratio: float = 0.0
    material_obj: Optional[Material] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "material_obj"}
        return d


@dataclass
class LayerDose:
    name: str
    thickness_um: float
    path_um: float
    density: float
    incident_flux: float
    transmitted_flux: float
    transmission: float
    mixed_dose_rate: float
    mixed_dose_rate_max: float
    components: List[ComponentDose]
    mixed_dose: Optional[float] = None
    mixed_dose_max: Optional[float] = None
    mixed_kerma_rate: float = 0.0
    mixed_kerma_rate_max: float = 0.0
    mixed_fluorescence_ratio: float = 0.0


class DoseResult:
    """Result of :meth:`Stack.compute`.

    Attributes
    ----------
    layers : list of LayerDose
    components : list of ComponentDose (flat)
    transmitted : Spectrum
        Spectrum leaving the stack.
    exposure_s : float or None
    """

    def __init__(self, stack: "Stack", beam: Beam, layers: List[LayerDose], spectra_in: List[Spectrum],
                 transmitted: Spectrum, deposition: str, electron_escape: bool,
                 exposure_s: Optional[float], _cache: dict):
        self.stack = stack
        self.beam = beam
        self.layers = layers
        self.spectra_in = spectra_in
        self.transmitted = transmitted
        self.deposition = deposition
        self.electron_escape = electron_escape
        self.exposure_s = exposure_s
        self._cache = _cache
        if exposure_s is not None:
            self._apply_exposure(exposure_s)

    def _apply_exposure(self, t: float) -> None:
        for L in self.layers:
            L.mixed_dose = L.mixed_dose_rate * t
            L.mixed_dose_max = L.mixed_dose_rate_max * t
            for c in L.components:
                c.dose = c.dose_rate * t
                c.dose_max = c.dose_rate_max * t

    def with_exposure(self, exposure_s: float) -> "DoseResult":
        """Same rates, doses for another exposure time (s)."""
        import copy
        new = copy.deepcopy(self)
        new.exposure_s = float(exposure_s)
        new._apply_exposure(new.exposure_s)
        return new

    @property
    def components(self) -> List[ComponentDose]:
        return [c for L in self.layers for c in L.components]

    @property
    def transmission(self) -> float:
        """Photon transmission of the whole stack."""
        return self.transmitted.total_flux / self.beam.spectrum.total_flux

    def get(self, layer: str, material: Optional[str] = None) -> Union[ComponentDose, LayerDose]:
        """Look up a layer (and optionally one of its components) by name."""
        for L in self.layers:
            if L.name == layer:
                if material is None:
                    return L
                for c in L.components:
                    if c.material == material:
                        return c
                raise KeyError(f"no material {material!r} in layer {layer!r}")
        raise KeyError(f"no layer {layer!r}")

    def table(self, unit: str = "kGy"):
        """Per-component table (pandas DataFrame if pandas is installed, else list of dicts).

        ``unit``: ``"Gy"``, ``"kGy"`` or ``"MGy"`` for doses and dose rates.
        """
        scale = {"gy": 1.0, "kgy": 1e-3, "mgy": 1e-6}[unit.lower()]
        rows = []
        for L in self.layers:
            for c in L.components:
                r = dict(layer=L.name, material=c.material, volume_fraction=c.volume_fraction,
                         interaction_pct=100 * c.interaction_fraction_layer,
                         energy_absorbed_pct=100 * c.energy_fraction,
                         **{f"dose_rate_{unit}_s": c.dose_rate * scale,
                            f"dose_rate_max_{unit}_s": c.dose_rate_max * scale,
                            f"mixed_dose_rate_{unit}_s": L.mixed_dose_rate * scale})
                if self.electron_escape:
                    r["retention"] = c.retention_mean
                if c.dose is not None:
                    r[f"dose_{unit}"] = c.dose * scale
                    r[f"dose_max_{unit}"] = c.dose_max * scale
                rows.append(r)
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:  # pragma: no cover
            return rows

    def summary(self, unit: str = "kGy") -> str:
        """Human-readable summary."""
        scale = {"gy": 1.0, "kgy": 1e-3, "mgy": 1e-6}[unit.lower()]
        b = self.beam
        s = b.spectrum
        en = (f"{s.energy_ev[0] / 1e3:.4g} keV" if s.is_monochromatic else
              f"{s.energy_ev[0] / 1e3:.4g}-{s.energy_ev[-1] / 1e3:.4g} keV, mean {s.mean_energy_ev / 1e3:.4g} keV")
        out = [f"Beam: {en}, {s.total_flux:.3e} ph/s, {s.power_w * 1e3:.3g} mW, "
               f"{b.width_um:g} x {b.height_um:g} um {b.profile}; incidence {self.stack.incidence_angle_deg:g} deg",
               f"Deposition: {self.deposition}" + (f"; photoelectron escape ON ({self.electron_escape}, detour "
                                                   f"{getattr(self, 'detour_factor', None)})"
                                                   if self.electron_escape else ""),
               f"Exposure: {self.exposure_s:g} s" if self.exposure_s is not None else "Exposure: (rates only)",
               ("Fluorescence reabsorption: " + self.fluorescence.summary()) if getattr(self, "fluorescence", None)
               else "Fluorescence reabsorption: off (mu_en: all fluorescence escapes)",
               f"Stack transmission: {100 * self.transmission:.2f} %"]
        for L in self.layers:
            out.append(f"- {L.name}: {L.thickness_um:g} um, T = {100 * L.transmission:.2f} %, "
                       f"mixed dose rate {L.mixed_dose_rate * scale:.4g} {unit}/s")
            for c in L.components:
                line = (f"    {c.material:<22s} v={c.volume_fraction:.3g}  interacts {100 * c.interaction_fraction_layer:.3g} %  "
                        f"dose rate {c.dose_rate * scale:.4g} {unit}/s (max {c.dose_rate_max * scale:.4g})")
                if c.dose is not None:
                    line += f"  dose {c.dose * scale:.4g} {unit}"
                if self.electron_escape:
                    line += f"  [retention {c.retention_mean:.2f}]"
                if c.fluorescence_ratio:
                    line += f"  [+{100 * c.fluorescence_ratio:.1f} % fluorescence]"
                out.append(line)
        return "\n".join(out)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def depth_profile(self, layer: str, material: Optional[str] = None, n: int = 101,
                      peak: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Dose rate (Gy/s) versus depth along the layer normal (um).

        ``material=None`` gives the mixed (layer) dose.  ``peak=True`` uses the
        peak fluence of the profile instead of the signal-weighted mean.
        Photoelectron-escape retention, when enabled, is applied as a constant
        over depth.  For a semi-infinite layer, the profile spans 5
        attenuation lengths at the mean energy.
        """
        j = [L.name for L in self.layers].index(layer)
        c = self._cache["layers"][j]
        L = self.layers[j]
        cosang = self._cache["cos"]
        t = L.thickness_um
        if not np.isfinite(t):
            e_mean = self.spectra_in[j].mean_energy_ev
            t = 5 * UM_PER_CM / float(self.stack.layers[j].mu(e_mean)[0]) * cosang
        z = np.linspace(0.0, t, n)
        s_cm = z / cosang / UM_PER_CM
        att = np.exp(-np.outer(s_cm, c["mu_star"]))             # (n, K)
        if material is None:
            dep, ret = c["dep_mixed"], c["ret_mixed_peak" if peak else "ret_mixed_mean"] + c["fl_mixed"]
        else:
            names = [m.name for m, _ in self.stack.layers[j].components]
            k = names.index(material)
            dep, ret = c["dep"][k], c["ret_peak" if peak else "ret_mean"][k] + c["fl"][k]
        f = self.beam.peak_factor if peak else self.beam.mean_factor
        fl = self.spectra_in[j].photons_per_s * f / self.beam.nominal_area_cm2
        rate = att @ (fl * self.spectra_in[j].energy_ev * J_PER_EV * dep * ret) * 1e3
        return z, rate

    def spectral_contribution(self, layer: str, material: Optional[str] = None,
                              peak: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Contribution of each spectral line to the dose rate (Gy/s per line).

        Returns ``(energy_ev, contribution)``; the contributions sum to
        ``dose_rate`` (or ``dose_rate_max`` with ``peak=True``, entrance face),
        or to the mixed layer value when ``material`` is None.  Divide by the
        bin widths (``Spectrum.density`` logic) for a per-eV density.
        """
        j = [L.name for L in self.layers].index(layer)
        c = self._cache["layers"][j]
        if material is None:
            dep, ret = c["dep_mixed"], c["ret_mixed_peak" if peak else "ret_mixed_mean"] + c["fl_mixed"]
        else:
            names = [m.name for m, _ in self.stack.layers[j].components]
            k = names.index(material)
            dep, ret = c["dep"][k], c["ret_peak" if peak else "ret_mean"][k] + c["fl"][k]
        f = self.beam.peak_factor if peak else self.beam.mean_factor
        s = self.spectra_in[j]
        att = 1.0 if peak else c["att_avg"]
        contrib = s.photons_per_s * f / self.beam.nominal_area_cm2 * s.energy_ev * J_PER_EV * dep * att * ret * 1e3
        return s.energy_ev.copy(), contrib

    def time_to_dose(self, dose_gy: float, layer: str, material: Optional[str] = None,
                     peak: bool = True) -> float:
        """Beam-on time (s) for a component (or the mixed layer) to reach ``dose_gy``.

        ``peak=True`` (default) uses the peak dose rate (beam centre, entrance
        face), the conservative choice for a damage threshold.
        """
        if material is None:
            L = self.get(layer)
            rate = L.mixed_dose_rate_max if peak else L.mixed_dose_rate
        else:
            c = self.get(layer, material)
            rate = c.dose_rate_max if peak else c.dose_rate
        return dose_gy / rate if rate > 0 else math.inf

    def plot(self, unit: str = "kGy", ax=None, what: str = "dose_rate"):
        """Bar chart of the per-component dose (or dose rate) and the mixed layer dose."""
        from .plotting import plot_result
        return plot_result(self, unit=unit, ax=ax, what=what)


class Stack:
    """An ordered list of layers (upstream first).

    Parameters
    ----------
    layers : list of Layer
    incidence_angle_deg : float
        Angle between the beam and the layer normal (0 = normal incidence).
    grazing_angle_deg : float, optional
        Alternative: angle between the beam and the surface (``90 - incidence``).
        BM32 micro-Laue: ~40 deg grazing.
    """

    def __init__(self, layers: Sequence[Layer], incidence_angle_deg: float = 0.0,
                 grazing_angle_deg: Optional[float] = None):
        layers = list(layers)
        if not layers:
            raise ValueError("empty stack")
        for L in layers:
            if not isinstance(L, Layer):
                raise TypeError("Stack takes Layer objects")
        names = [L.name for L in layers]
        if len(set(names)) != len(names):
            raise ValueError("layer names must be unique (they are used to look results up)")
        if grazing_angle_deg is not None:
            incidence_angle_deg = 90.0 - float(grazing_angle_deg)
        if not 0.0 <= incidence_angle_deg < 89.9:
            raise ValueError("incidence angle must be in [0, 89.9) deg from the normal")
        for L in layers[:-1]:
            if not np.isfinite(L.thickness_um):
                raise ValueError("only the last layer can be semi-infinite")
        self.layers = layers
        self.incidence_angle_deg = float(incidence_angle_deg)

    def compute(self, beam: Beam, exposure_s: Optional[float] = None,
                deposition: str = "energy_absorption", electron_escape=False,
                detour_factor: Optional[float] = _escape.DEFAULT_DETOUR_FACTOR, fluorescence: bool = False,
                mc_photons: int = 20000, mc_electrons: int = 1500, seed: int = 12345) -> DoseResult:
        """Propagate ``beam`` through the stack and compute the doses.

        Parameters
        ----------
        beam : Beam
        exposure_s : float, optional
            Accumulated exposure; doses are then reported along with the rates.
        deposition : {"energy_absorption", "attenuation"}
            ``"energy_absorption"`` (default): energy deposited = mu_en.
            ``"attenuation"``: every interacting photon deposits all its energy
            locally (the 2023 model; an upper bound that includes scattered and
            fluorescence energy).
        electron_escape : bool or {"analytic", "mc"}
            Photoelectron escape from the beam footprint (relevant when the
            beam is not much larger than a few um).  ``True``/``"analytic"``:
            straight-track estimate of :mod:`escape` (fast).  ``"mc"``:
            electron Monte Carlo of :mod:`electron_mc` on a few energies of the
            spectrum, with a free surface for a semi-infinite last layer
            (slower: seconds per component).  The kerma (no escape) is always
            reported as ``kerma_rate``.
        detour_factor : float or None
            Analytic escape only: effective reach / Kanaya-Okayama range.
            ``None`` (default) = Z-dependent value fitted to the Monte Carlo;
            0.4 reproduces micro-beam *damage* ratios measured on protein
            crystals, which are smaller than the physical escape.
        fluorescence : bool
            Add the fluorescence reabsorbed inside the beam footprint
            (:mod:`fluorescence`, Monte Carlo through the stack).  Matters for
            beams larger than the fluorescence attenuation length (tens of um
            for 3d-metal K lines).  Only with ``deposition="energy_absorption"``.
        mc_photons, seed : int
            Photons per fluorescence line source, and the random seed.
        """
        if deposition not in ("energy_absorption", "attenuation"):
            raise ValueError("deposition must be 'energy_absorption' or 'attenuation'")
        if electron_escape is True:
            electron_escape = "analytic"
        if electron_escape not in (False, None, "analytic", "mc"):
            raise ValueError("electron_escape must be False, True, 'analytic' or 'mc'")
        if fluorescence and deposition != "energy_absorption":
            raise ValueError("fluorescence reabsorption only applies to deposition='energy_absorption' "
                             "('attenuation' already deposits all fluorescence locally)")
        spec = beam.spectrum
        e = spec.energy_ev
        e_j = e * J_PER_EV
        cosang = math.cos(math.radians(self.incidence_angle_deg))
        area = beam.nominal_area_cm2
        n_in = spec.photons_per_s.copy()
        n0_tot = spec.total_flux
        p0 = spec.power_w
        layer_results, spectra_in, cache_layers = [], [], []
        for L in self.layers:
            spectra_in.append(Spectrum(e, n_in.copy(), label=f"into {L.name}"))
            path_um = L.thickness_um / cosang
            l_cm = path_um / UM_PER_CM
            mus = [m.mu(e) for m, _ in L.components]                       # 1/cm
            mu_star = sum(v * mu for (m, v), mu in zip(L.components, mus))
            if deposition == "energy_absorption":
                deps = [m.mu_en_rho(e) for m, _ in L.components]          # cm2/g
            else:
                deps = [m.mu_rho(e) for m, _ in L.components]
            if np.isfinite(l_cm):
                x = mu_star * l_cm
                absorbed = -np.expm1(-x)                                   # 1 - exp(-x)
                with np.errstate(invalid="ignore", divide="ignore"):
                    att_avg = np.where(x > 1e-12, absorbed / x, 1.0 - x / 2)
                n_out = n_in * np.exp(-x)
            else:
                absorbed = np.ones_like(e)
                att_avg = np.full_like(e, 1.0 - math.exp(-1.0))           # average over 1 att. length
                n_out = np.zeros_like(e)
            # escape
            rho_l = L.density
            w_l = L.mass_fractions
            dep_mixed = sum(v * m.density * d for (m, v), d in zip(L.components, deps)) / rho_l
            if electron_escape == "analytic":
                rets = [_escape.component_retention(m.mass_fractions, m.mass_fractions, m.density, e,
                                                    beam.width_um, beam.height_um, beam.profile,
                                                    detour_factor)
                        for m, _ in L.components]
                ret_mixed = _escape.component_retention(w_l, w_l, rho_l, e, beam.width_um,
                                                        beam.height_um, beam.profile, detour_factor)
            elif electron_escape == "mc":
                from . import electron_mc as _emc
                graz = (90.0 - self.incidence_angle_deg) if not np.isfinite(l_cm) else None
                wts = n_in * e_j * att_avg
                rets = [_emc.component_retention_mc(m.mass_fractions, m.density, e, beam.width_um,
                                                    beam.height_um, beam.profile, graz,
                                                    n_electrons=mc_electrons, seed=seed + 31 * c_i,
                                                    weights=wts * m.mu_en_rho(e))
                        for c_i, (m, _) in enumerate(L.components)]
                ret_mixed = _emc.component_retention_mc(w_l, rho_l, e, beam.width_um, beam.height_um,
                                                        beam.profile, graz, n_electrons=mc_electrons,
                                                        seed=seed + 997, weights=wts)
            else:
                ones = np.ones_like(e)
                rets = [(ones, ones) for _ in L.components]
                ret_mixed = (ones, ones)
            fl_mean = n_in * beam.mean_factor / area                        # ph/s/cm2 per line
            fl_peak = n_in * beam.peak_factor / area
            comp_results = []
            for (m, v), mu, d, (rm, rp) in zip(L.components, mus, deps, rets):
                kerma = float(np.sum(fl_mean * e_j * d * att_avg)) * 1e3     # J/g/s -> Gy/s
                kerma_max = float(np.sum(fl_peak * e_j * d)) * 1e3
                rate = float(np.sum(fl_mean * e_j * d * att_avg * rm)) * 1e3
                rate_max = float(np.sum(fl_peak * e_j * d * rp)) * 1e3
                with np.errstate(invalid="ignore", divide="ignore"):
                    share = np.where(mu_star > 0, v * mu / mu_star, 0.0)
                    dshare = np.where(mu_star > 0, v * m.density * d / mu_star, 0.0)
                n_int = float(np.sum(n_in * share * absorbed))
                power = float(np.sum(n_in * e_j * dshare * absorbed))
                mass = m.density * v * area * l_cm if np.isfinite(l_cm) else None
                comp_results.append(ComponentDose(
                    layer=L.name, material=m.name, volume_fraction=v, density=m.density, mass_g=mass,
                    interaction_fraction=n_int / n0_tot,
                    interaction_fraction_layer=n_int / max(float(n_in.sum()), 1e-300),
                    energy_fraction=power / p0 if p0 > 0 else 0.0, absorbed_power_w=power,
                    kerma_rate=kerma, kerma_rate_max=kerma_max,
                    retention_mean=rate / kerma if kerma > 0 else 1.0,
                    retention_peak=rate_max / kerma_max if kerma_max > 0 else 1.0,
                    dose_rate=rate, dose_rate_max=rate_max, material_obj=m))
            mixed = float(np.sum(fl_mean * e_j * dep_mixed * att_avg * ret_mixed[0])) * 1e3
            mixed_max = float(np.sum(fl_peak * e_j * dep_mixed * ret_mixed[1])) * 1e3
            mixed_kerma = float(np.sum(fl_mean * e_j * dep_mixed * att_avg)) * 1e3
            mixed_kerma_max = float(np.sum(fl_peak * e_j * dep_mixed)) * 1e3
            layer_results.append(LayerDose(
                name=L.name, thickness_um=L.thickness_um, path_um=path_um, density=rho_l,
                incident_flux=float(n_in.sum()), transmitted_flux=float(n_out.sum()),
                transmission=float(n_out.sum() / max(n_in.sum(), 1e-300)),
                mixed_dose_rate=mixed, mixed_dose_rate_max=mixed_max, components=comp_results,
                mixed_kerma_rate=mixed_kerma, mixed_kerma_rate_max=mixed_kerma_max))
            cache_layers.append(dict(mu_star=mu_star, dep=deps, dep_mixed=dep_mixed, att_avg=att_avg,
                                     ret_mean=[r[0] for r in rets], ret_peak=[r[1] for r in rets],
                                     ret_mixed_mean=ret_mixed[0], ret_mixed_peak=ret_mixed[1]))
            n_in = n_out
        transmitted = Spectrum(e, n_in, label="transmitted")
        fl_report = None
        for cl, L in zip(cache_layers, layer_results):
            cl["fl"] = [0.0] * len(L.components)
            cl["fl_mixed"] = 0.0
        if fluorescence:
            primary = {(j, c): comp.absorbed_power_w
                       for j, LR in enumerate(layer_results) for c, comp in enumerate(LR.components)}
            fl_report = _fluo.reabsorption(self, beam, spectra_in, primary, cosang,
                                           n_photons=mc_photons, seed=seed)
            for j, (LR, cl) in enumerate(zip(layer_results, cache_layers)):
                for c, comp in enumerate(LR.components):
                    r = fl_report.ratios.get((j, c), 0.0)
                    comp.fluorescence_ratio = r
                    comp.dose_rate += r * comp.kerma_rate
                    comp.dose_rate_max += r * comp.kerma_rate_max
                    cl["fl"][c] = r
                r = fl_report.mixed_ratios.get(j, 0.0)
                LR.mixed_fluorescence_ratio = r
                LR.mixed_dose_rate += r * LR.mixed_kerma_rate
                LR.mixed_dose_rate_max += r * LR.mixed_kerma_rate_max
                cl["fl_mixed"] = r
        res = DoseResult(self, beam, layer_results, spectra_in, transmitted, deposition,
                         electron_escape, exposure_s, dict(layers=cache_layers, cos=cosang))
        res.detour_factor = (("Z-dependent" if detour_factor is None else detour_factor)
                             if electron_escape == "analytic" else ("Monte Carlo" if electron_escape else None))
        res.fluorescence = fl_report
        return res

    def transmission(self, energy_ev) -> np.ndarray:
        """Photon transmission of the stack versus energy."""
        e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
        cosang = math.cos(math.radians(self.incidence_angle_deg))
        x = sum(L.mu(e) * L.thickness_um / cosang / UM_PER_CM for L in self.layers)
        return np.exp(-x)

    def __repr__(self) -> str:
        return "Stack([" + ", ".join(L.name for L in self.layers) + f"], incidence={self.incidence_angle_deg:g} deg)"
