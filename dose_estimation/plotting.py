"""Matplotlib helpers (imported lazily; matplotlib is optional)."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["plot_result", "plot_stack", "plot_depth_profiles", "plot_spectrum"]

_SCALE = {"gy": 1.0, "kgy": 1e-3, "mgy": 1e-6}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_result(result, unit: str = "kGy", ax=None, what: str = "dose_rate", log: bool = True):
    """Horizontal bars: per-component dose (or dose rate), with the mixed layer value as a tick.

    ``what``: ``"dose_rate"`` or ``"dose"`` (needs an exposure).
    """
    plt = _plt()
    s = _SCALE[unit.lower()]
    if what == "dose" and result.exposure_s is None:
        raise ValueError("no exposure: compute(..., exposure_s=...) or result.with_exposure(t)")
    labels, vals, vmax, mixed = [], [], [], []
    for L in result.layers:
        for c in L.components:
            labels.append(f"{L.name} / {c.material}")
            if what == "dose":
                vals.append(c.dose * s); vmax.append(c.dose_max * s); mixed.append(L.mixed_dose * s)
            else:
                vals.append(c.dose_rate * s); vmax.append(c.dose_rate_max * s); mixed.append(L.mixed_dose_rate * s)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color="0.6", label="per component (depth average)")
    ax.plot(vmax, y, "|", color="k", ms=12, label="peak (entrance, beam centre)")
    ax.plot(mixed, y, "o", mfc="none", color="C0", label="layer, electrons shared")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(f"{'dose' if what == 'dose' else 'dose rate'} ({unit}{'' if what == 'dose' else '/s'})")
    ax.legend(loc="lower right", fontsize="small")
    ax.set_title(f"{result.deposition}" + (", photoelectron escape" if result.electron_escape else ""),
                 fontsize="small")
    return ax


def plot_stack(result, ax=None, unit: str = "kGy"):
    """The v0 figure: the stack drawn to scale along the beam, annotated with the doses."""
    plt = _plt()
    s = _SCALE[unit.lower()]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    z0 = 0.0
    for i, L in enumerate(result.layers):
        t = L.thickness_um if np.isfinite(L.thickness_um) else 0.2 * max(z0, 1.0)
        solid_frac = L.components[0].volume_fraction
        ax.add_patch(plt.Rectangle((z0, 0), t, 1, facecolor=colors[i % len(colors)],
                                   alpha=0.25 + 0.75 * solid_frac))
        txt = [f"{L.name} ({L.thickness_um:g} um, T={100 * L.transmission:.1f}%)"]
        for c in L.components:
            d = c.dose if c.dose is not None else c.dose_rate
            txt.append(f" {c.material}: {d * s:.3g} {unit}{'' if c.dose is not None else '/s'}")
        ax.text(z0, 0.75 + 0.2 * (-1) ** i, "\n".join(txt), va="top", fontsize="x-small")
        z0 += t
    ax.set_xlim(0, z0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("depth along the layer normal (um)")
    return ax


def plot_depth_profiles(result, unit: str = "kGy", ax=None, peak: bool = False, n: int = 60):
    """Dose rate versus depth through the whole stack, one line per component."""
    plt = _plt()
    s = _SCALE[unit.lower()]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    names = list(dict.fromkeys(c.material for L in result.layers for c in L.components))
    color = {m: f"C{i % 10}" for i, m in enumerate(names)}
    seen = set()
    z0 = 0.0
    for L in result.layers:
        for c in L.components:
            z, r = result.depth_profile(L.name, c.material, n=n, peak=peak)
            ax.plot(z0 + z, r * s, color=color[c.material],
                    label=None if c.material in seen else c.material)
            seen.add(c.material)
        z0 += z[-1]
        ax.axvline(z0, color="0.8", lw=0.8)
        ax.text(z0 - z[-1] / 2, 1.0, L.name, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize="x-small")
    ax.set_yscale("log")
    ax.set_xlabel("depth along the layer normal (um)")
    ax.set_ylabel(f"dose rate ({unit}/s)")
    ax.legend(fontsize="small", loc="best")
    return ax


def plot_spectrum(spectrum, ax=None, label: Optional[str] = None, per: str = "eV", **kw):
    """Spectral density (photons/s/eV, or /keV) of a white beam; a stem for lines."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    e = spectrum.energy_ev / 1e3
    if spectrum.energy_ev.size < 3:
        ax.vlines(e, 0, spectrum.photons_per_s, label=label or spectrum.label, **kw)
        ax.set_ylabel("photons/s")
    else:
        f = 1e3 if per.lower() == "kev" else 1.0
        ax.plot(e, spectrum.density() * f, label=label or spectrum.label, **kw)
        ax.set_ylabel(f"photons/s/{per}")
    ax.set_xlabel("energy (keV)")
    return ax
