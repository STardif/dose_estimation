"""Materials: composition, density and X-ray coefficients.

A :class:`Material` is a homogeneous substance described by its elemental
**mass fractions** and its density.  It can be built from

* a chemical formula, ``Material("LiNi0.8Mn0.1Co0.1O2", 4.8)``; parentheses and
  hydrates are understood (``"Li(Ni0.8Mn0.1Co0.1)O2"``, ``"CuSO4*5H2O"``);
* a mixture by mass, :meth:`Material.from_mass_fractions` (e.g. an electrode
  coating 96 % NMC / 2.5 % PVDF / 1.5 % carbon);
* a mixture by volume, :meth:`Material.from_volume_fractions` (e.g. a solvent
  blend EC:PC:DMC 1:1:3);
* a solution of known molarity, :meth:`Material.solution` (e.g. 1 M LiPF6).

The mixture rule is used for the coefficients,
``(mu/rho)_mix = sum_i w_i (mu/rho)_i`` and likewise for ``mu_en/rho``.  It is
exact for ``mu/rho`` and an excellent approximation for ``mu_en/rho`` in the
X-ray range (the only neglected term is bremsstrahlung of the secondary
electrons, negligible below ~1 MeV).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Union

import numpy as np

from . import xray_data as xd
from .constants import UM_PER_CM

__all__ = ["Material", "parse_formula"]

_TOKEN = re.compile(r"([A-Z][a-z]?)|(\()|(\))|(\d*\.\d+|\d+\.?)")


def parse_formula(formula: str) -> "OrderedDict[str, float]":
    """Parse a chemical formula into ``{element: atoms per formula unit}``.

    Supports decimal stoichiometry (``LiNi0.8Mn0.1Co0.1O2``, ``Co.4``),
    nested parentheses (``Li(Ni0.8Mn0.1Co0.1)O2``, ``(C2H2F2)``) and hydrates
    or adducts joined by ``*`` or a middle dot (``CuSO4*5H2O``).
    Whitespace is ignored.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty string")
    total: "OrderedDict[str, float]" = OrderedDict()
    for part in re.split(r"[*·•]", formula.replace(" ", "")):
        if not part:
            raise ValueError(f"Cannot parse formula {formula!r}")
        m = re.match(r"^(\d*\.?\d+)(?=[A-Z(])", part)       # leading multiplier, e.g. 5H2O
        mult = 1.0
        if m:
            mult = float(m.group(1))
            part = part[m.end():]
        for el, n in _parse_group(part, formula).items():
            total[el] = total.get(el, 0.0) + mult * n
    for el in total:
        xd.atomic_number(el)                                 # validates the symbol
    return total


def _parse_group(s: str, original: str) -> "OrderedDict[str, float]":
    """Tokenise and accumulate; a number multiplies the element or group just before it."""
    pos = 0
    stack = [OrderedDict()]
    pending = None       # ("el", symbol) or ("grp", dict): waiting for an optional count

    def flush(mult: float = 1.0):
        nonlocal pending
        if pending is None:
            return
        kind, ref = pending
        d = stack[-1]
        if kind == "el":
            d[ref] = d.get(ref, 0.0) + mult
        else:
            for k, v in ref.items():
                d[k] = d.get(k, 0.0) + v * mult
        pending = None

    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError(f"Cannot parse formula {original!r} near {s[pos:]!r}")
        el, lpar, rpar, num = m.groups()
        pos = m.end()
        if el:
            flush()
            pending = ("el", el)
        elif lpar:
            flush()
            stack.append(OrderedDict())
        elif rpar:
            flush()
            if len(stack) == 1:
                raise ValueError(f"Unbalanced parenthesis in {original!r}")
            pending = ("grp", stack.pop())
        else:
            if pending is None:
                raise ValueError(f"Misplaced number in {original!r}")
            flush(float(num))
    flush()
    if len(stack) != 1:
        raise ValueError(f"Unbalanced parenthesis in {original!r}")
    return stack[0]


MaterialLike = Union["Material", str]


class Material:
    """A homogeneous material.

    Parameters
    ----------
    formula : str, optional
        Chemical formula (atoms per formula unit).  Either ``formula`` or
        ``mass_fractions`` must be given.
    density : float
        Density in g/cm^3.
    name : str, optional
        Label used in the results (defaults to the formula).
    mass_fractions : mapping, optional
        ``{element symbol: mass fraction}``; normalised to 1.

    Notes
    -----
    A ``Material`` holds no computation results: the same instance can be
    used in several layers (e.g. one electrolyte everywhere in a cell).
    """

    def __init__(self, formula: Optional[str] = None, density: Optional[float] = None,
                 name: Optional[str] = None, *, mass_fractions: Optional[Mapping[str, float]] = None):
        if density is None or not np.isfinite(density) or density <= 0:
            raise ValueError("density must be a positive number (g/cm^3)")
        self.density = float(density)
        if (formula is None) == (mass_fractions is None):
            raise ValueError("give exactly one of formula or mass_fractions")
        self.formula = formula
        if formula is not None:
            self.atoms = parse_formula(formula)
            masses = {el: n * xd.atomic_weight(el) for el, n in self.atoms.items()}
            self.molar_mass = sum(masses.values())
            w = {el: m / self.molar_mass for el, m in masses.items()}
        else:
            self.atoms = None
            self.molar_mass = None
            w = {}
            for el, f in mass_fractions.items():
                xd.atomic_number(el)
                if f < 0:
                    raise ValueError("mass fractions must be >= 0")
                w[el] = w.get(el, 0.0) + float(f)
            s = sum(w.values())
            if s <= 0:
                raise ValueError("mass fractions sum to zero")
            w = {el: f / s for el, f in w.items()}
        self.mass_fractions: Dict[str, float] = dict(w)
        self.name = name or formula or "mixture"

    # -- constructors --------------------------------------------------------
    @staticmethod
    def _as_material(m: MaterialLike, density: Optional[float] = None) -> "Material":
        if isinstance(m, Material):
            return m
        return Material(m, density if density is not None else 1.0)

    @classmethod
    def from_mass_fractions(cls, parts: Mapping[MaterialLike, float], density: Optional[float] = None,
                            name: Optional[str] = None) -> "Material":
        """Mixture by mass.

        ``parts`` maps :class:`Material` instances (or formulas) to mass
        fractions (normalised to 1).  If ``density`` is omitted it is computed
        for ideal mixing, ``1/rho = sum_i w_i / rho_i`` (formula strings then
        cannot be used, as they carry no density).
        """
        tot = float(sum(parts.values()))
        if tot <= 0:
            raise ValueError("fractions sum to zero")
        w_el: Dict[str, float] = {}
        inv_rho = 0.0
        for m, f in parts.items():
            mat = cls._as_material(m)
            if density is None and not isinstance(m, Material):
                raise ValueError("density is required when parts are given as formulas")
            for el, we in mat.mass_fractions.items():
                w_el[el] = w_el.get(el, 0.0) + we * f / tot
            inv_rho += (f / tot) / mat.density
        rho = density if density is not None else 1.0 / inv_rho
        label = name or " + ".join(f"{100 * f / tot:.3g}% {cls._as_material(m).name}" for m, f in parts.items())
        return cls(density=rho, name=label, mass_fractions=w_el)

    @classmethod
    def from_volume_fractions(cls, parts: Mapping["Material", float], name: Optional[str] = None,
                              density: Optional[float] = None) -> "Material":
        """Mixture by volume (ideal mixing unless ``density`` is given).

        ``parts`` maps :class:`Material` instances to volume fractions
        (normalised to 1).  Example: EC:PC:DMC 1:1:3 is
        ``{ec: 1, pc: 1, dmc: 3}``.
        """
        tot = float(sum(parts.values()))
        masses = {m: (v / tot) * m.density for m, v in parts.items()}
        rho_ideal = sum(masses.values())
        w = {m: mm / rho_ideal for m, mm in masses.items()}
        label = name or " : ".join(f"{m.name}" for m in parts) + " (vol. " + \
            ":".join(f"{v:g}" for v in parts.values()) + ")"
        return cls.from_mass_fractions(w, density=density if density is not None else rho_ideal, name=label)

    @classmethod
    def solution(cls, solvent: "Material", solute: MaterialLike, molarity: float, density: float,
                 name: Optional[str] = None) -> "Material":
        """Solution of ``molarity`` mol/L of ``solute`` in ``solvent``.

        ``density`` is the density of the *solution* (g/cm^3), which sets the
        mass of solvent per litre.  Example: 1 M LiPF6 in EC/DMC,
        ``Material.solution(ecdmc, "LiPF6", 1.0, density=1.27)``.
        """
        sol = solute if isinstance(solute, Material) else Material(solute, 1.0)
        if sol.molar_mass is None:
            raise ValueError("the solute must be defined by a formula (its molar mass is needed)")
        m_solute = molarity * sol.molar_mass            # g per litre of solution
        m_total = density * 1000.0                      # g per litre of solution
        if m_solute >= m_total:
            raise ValueError("solute mass exceeds the solution mass: check molarity/density")
        w = (m_solute / m_total)
        w_el: Dict[str, float] = {}
        for el, f in solvent.mass_fractions.items():
            w_el[el] = w_el.get(el, 0.0) + f * (1 - w)
        for el, f in sol.mass_fractions.items():
            w_el[el] = w_el.get(el, 0.0) + f * w
        label = name or f"{molarity:g} M {sol.name} in {solvent.name}"
        return cls(density=density, name=label, mass_fractions=w_el)

    def with_density(self, density: float, name: Optional[str] = None) -> "Material":
        """Same composition, another density (e.g. a porous or compacted form)."""
        return Material(density=density, name=name or self.name, mass_fractions=self.mass_fractions)

    # -- coefficients ----------------------------------------------------------
    def mu_rho(self, energy_ev) -> np.ndarray:
        """Mass attenuation coefficient, cm^2/g."""
        e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
        return sum(w * xd.mu_rho(el, e) for el, w in self.mass_fractions.items())

    def mu_en_rho(self, energy_ev) -> np.ndarray:
        """Mass energy-absorption coefficient, cm^2/g."""
        e = np.atleast_1d(np.asarray(energy_ev, dtype=float))
        return sum(w * xd.mu_en_rho(el, e) for el, w in self.mass_fractions.items())

    def mu(self, energy_ev) -> np.ndarray:
        """Linear attenuation coefficient, 1/cm."""
        return self.density * self.mu_rho(energy_ev)

    def mu_en(self, energy_ev) -> np.ndarray:
        """Linear energy-absorption coefficient, 1/cm."""
        return self.density * self.mu_en_rho(energy_ev)

    def attenuation_length_um(self, energy_ev) -> np.ndarray:
        """1/e attenuation length, um."""
        return UM_PER_CM / self.mu(energy_ev)

    def transmission(self, energy_ev, thickness_um: float) -> np.ndarray:
        """Beer-Lambert transmission of a slab of this material."""
        return np.exp(-self.mu(energy_ev) * thickness_um / UM_PER_CM)

    # -- legacy (v0) names -----------------------------------------------------
    def massAbsCoeff(self, energy):  # noqa: N802 - v0 API
        """v0 name: now returns the NIST mass *attenuation* coefficient, cm^2/g."""
        return self.mu_rho(energy)

    def linAbsCoeff(self, energy):  # noqa: N802
        """v0 name: linear attenuation coefficient, 1/cm."""
        return self.mu(energy)

    def attenLength(self, energy):  # noqa: N802
        """v0 name: attenuation length, um."""
        return self.attenuation_length_um(energy)

    # -- misc --------------------------------------------------------------------
    def describe(self) -> str:
        comp = ", ".join(f"{el} {100 * w:.2f}%" for el, w in
                         sorted(self.mass_fractions.items(), key=lambda kv: -kv[1]))
        return f"{self.name}: {self.density:g} g/cm3; mass fractions {comp}"

    def __repr__(self) -> str:
        return f"Material(name={self.name!r}, density={self.density:g})"
