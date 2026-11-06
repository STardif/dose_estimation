"""Atomic relaxation data: edges, jump ratios, fluorescence yields, emission lines.

Bundled from XrayDB (https://github.com/xraypy/XrayDB, release 9.2, CC0 public
domain), which compiles the tables of Elam, Ravel & Sieber, Radiat. Phys.
Chem. 63, 121 (2002): absorption-edge energies, jump ratios and fluorescence
yields for the K, L1-L3 and M1-M5 levels, Coster-Kronig probabilities for the
L subshells, and emission-line energies and relative intensities.

These data serve two purposes:

* split the photo-absorption of an element between its shells (jump-ratio
  approximation) and turn the vacancies into fluorescence lines, for the
  fluorescence-reabsorption model (:mod:`~dose_estimation.fluorescence`);
* give the photo- and Auger-electron energies used by the electron-escape
  estimate (:mod:`~dose_estimation.escape`).
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["levels", "lines", "line_groups", "coster_kronig", "shell_photo_fractions",
           "fluorescence_lines", "fluorescence_energy_per_absorption", "binding_energy_below",
           "fluorescence_yield"]

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_SHELL_ORDER = ("K", "L1", "L2", "L3")


@lru_cache(maxsize=None)
def _levels_table() -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    with open(os.path.join(_DATA, "xray_levels.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["element"], {})[r["level"]] = (
                float(r["edge_eV"]), float(r["fluorescence_yield"]), float(r["jump_ratio"]))
    return out


@lru_cache(maxsize=None)
def _lines_table() -> Dict[str, Dict[str, List[Tuple[float, float, str]]]]:
    out: Dict[str, Dict[str, List[Tuple[float, float, str]]]] = {}
    with open(os.path.join(_DATA, "xray_lines.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["element"], {}).setdefault(r["initial"], []).append(
                (float(r["energy_eV"]), float(r["intensity"]), r["siegbahn"]))
    return out


@lru_cache(maxsize=None)
def _ck_table() -> Dict[str, Dict[Tuple[str, str], float]]:
    out: Dict[str, Dict[Tuple[str, str], float]] = {}
    with open(os.path.join(_DATA, "coster_kronig.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["element"], {})[(r["initial"], r["final"])] = float(r["probability"])
    return out


def levels(symbol: str) -> Dict[str, Tuple[float, float, float]]:
    """``{level: (edge_eV, fluorescence_yield, jump_ratio)}`` (empty for H, He)."""
    return dict(_levels_table().get(symbol, {}))


def lines(symbol: str, initial: str) -> List[Tuple[float, float, str]]:
    """Emission lines filling a vacancy in ``initial``: ``[(energy_eV, rel. intensity, name)]``."""
    return list(_lines_table().get(symbol, {}).get(initial, []))


def line_groups(symbol: str, initial: str, rel_tol: float = 0.02) -> List[Tuple[float, float]]:
    """Lines of a level merged when within ``rel_tol`` in energy (e.g. Ka1 + Ka2).

    Returns ``[(mean energy eV, summed intensity)]`` with intensities normalised to 1.
    """
    ls = sorted((e, i) for e, i, _ in lines(symbol, initial) if i > 0)
    if not ls:
        return []
    groups: List[List[float]] = []
    for e, i in ls:
        if groups and abs(e - groups[-1][0] / groups[-1][1]) <= rel_tol * e:
            groups[-1][0] += e * i
            groups[-1][1] += i
        else:
            groups.append([e * i, i])
    tot = sum(g[1] for g in groups)
    return [(g[0] / g[1], g[1] / tot) for g in groups]


def coster_kronig(symbol: str) -> Dict[Tuple[str, str], float]:
    """L-subshell Coster-Kronig probabilities ``{(initial, final): f}``."""
    return dict(_ck_table().get(symbol, {}))


def fluorescence_yield(symbol: str, level: str) -> float:
    lv = levels(symbol).get(level)
    return lv[1] if lv else 0.0


def shell_photo_fractions(symbol: str, energy_ev: float) -> Dict[str, float]:
    """Probability that a photo-absorption at ``energy_ev`` ionises each of K, L1, L2, L3.

    Jump-ratio approximation: above an edge of jump ratio J, a fraction
    (J - 1)/J of the photo-absorption happens on that shell, the remainder on
    the outer shells.
    """
    lv = levels(symbol)
    out: Dict[str, float] = {}
    remaining = 1.0
    for s in _SHELL_ORDER:
        if s not in lv:
            continue
        edge, _, jump = lv[s]
        if energy_ev > edge and jump > 1.0:
            p = remaining * (jump - 1.0) / jump
            out[s] = p
            remaining -= p
    return out


def _open_shells(symbol: str, energy_ev: float) -> Tuple[str, ...]:
    lv = levels(symbol)
    return tuple(s for s in _SHELL_ORDER if s in lv and energy_ev > lv[s][0])


@lru_cache(maxsize=None)
def _lines_for(symbol: str, open_shells: Tuple[str, ...]) -> Tuple[Tuple[float, float], ...]:
    lv = levels(symbol)
    p: Dict[str, float] = {}
    remaining = 1.0
    for s in open_shells:
        jump = lv[s][2]
        if jump > 1.0:
            p[s] = remaining * (jump - 1.0) / jump
            remaining -= p[s]
    if not p:
        return ()
    ck = coster_kronig(symbol)
    n = {s: p.get(s, 0.0) for s in _SHELL_ORDER}
    n["L2"] += ck.get(("L1", "L2"), 0.0) * n["L1"]
    n["L3"] += ck.get(("L1", "L3"), 0.0) * n["L1"] + ck.get(("L2", "L3"), 0.0) * n["L2"]
    out: List[Tuple[float, float]] = []
    for s in _SHELL_ORDER:
        if n[s] <= 0:
            continue
        w = fluorescence_yield(symbol, s)
        if w <= 0:
            continue
        for e, frac in line_groups(symbol, s):
            out.append((e, n[s] * w * frac))
    return tuple(out)


def fluorescence_lines(symbol: str, energy_ev: float) -> List[Tuple[float, float]]:
    """Fluorescence photons emitted per photo-absorption at ``energy_ev``.

    Returns ``[(line energy eV, photons per absorption)]`` for the K and L
    lines (merged within 2 %), with Coster-Kronig transfer between the L
    subshells.  Cascade vacancies (L vacancies created by K emission or
    K-LL Auger) are not followed: their lines are soft and their yields low.
    In the jump-ratio approximation the result depends on the photon energy
    only through the set of open shells, so it is cached per set.
    """
    return list(_lines_for(symbol, _open_shells(symbol, energy_ev)))


def fluorescence_energy_per_absorption(symbol: str, energy_ev: float) -> float:
    """Mean fluorescence energy X (eV) emitted per photo-absorption (``mu_en = tau (1 - X/E)``)."""
    return float(sum(e * n for e, n in _lines_for(symbol, _open_shells(symbol, energy_ev))))


def binding_energy_below(symbol: str, energy_ev: float) -> Tuple[float, str]:
    """Deepest level (K, L1-3, M1-5) whose edge lies below ``energy_ev``: ``(edge eV, level)``.

    This is the shell most photo-absorptions ionise.  ``(0, "")`` if none
    (H, He, or a photon below every edge).
    """
    best = (0.0, "")
    for lev, (edge, _, _) in levels(symbol).items():
        if edge < energy_ev and edge > best[0]:
            best = (edge, lev)
    return best
