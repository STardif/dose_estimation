"""Incident beam: spectrum, size and profile.

A :class:`Spectrum` is a set of discrete energies ``E_k`` (eV) carrying
``N_k`` photons/s each.  A monochromatic beam is one line; a white beam is a
fine grid whose ``N_k`` are the spectral density integrated over each bin.
Every dose integral is then a plain sum over k, which is exactly the
trapezoidal rule ``int dE (dN/dE) E mu_en(E)/rho`` for a sampled spectrum.

A :class:`Beam` adds the transverse size and profile, which set the fluence
(photons per unit area).

Conventions
-----------
* Energies in eV, flux in photons/s, sizes in um (``width`` is horizontal,
  ``height`` vertical).
* ``profile="tophat"``: uniform over ``width x height``; peak and mean fluence
  are both ``N / (w h)`` (the convention of the 2023 paper).
* ``profile="gaussian"``: ``width`` and ``height`` are FWHM.  Peak fluence
  ``N 4 ln2 / (pi w h) = 0.883 N/(w h)``; the *signal-weighted* mean fluence,
  ``int Phi^2 dA / int Phi dA``, is half the peak, ``0.441 N/(w h)``.  The
  latter is the dose "seen" by a measurement whose signal is proportional to
  the local fluence (diffraction, fluorescence), the 2D analogue of the
  diffraction-weighted dose of RADDOSE-3D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from .constants import CM2_PER_UM2, J_PER_EV

__all__ = ["Spectrum", "Beam"]


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Weights w such that sum(w * f) is the trapezoidal integral of f over x."""
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.ones(1)
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


@dataclass(frozen=True)
class Spectrum:
    """Photon spectrum as discrete lines.

    Parameters
    ----------
    energy_ev : array
        Photon energies, eV (strictly increasing).
    photons_per_s : array
        Photons/s carried by each line (or bin).
    label : str
    """

    energy_ev: np.ndarray
    photons_per_s: np.ndarray
    label: str = ""

    def __post_init__(self):
        e = np.atleast_1d(np.asarray(self.energy_ev, dtype=float))
        n = np.atleast_1d(np.asarray(self.photons_per_s, dtype=float))
        if e.shape != n.shape:
            raise ValueError("energy_ev and photons_per_s must have the same shape")
        if e.size == 0:
            raise ValueError("empty spectrum")
        if np.any(e <= 0) or np.any(~np.isfinite(e)):
            raise ValueError("energies must be positive")
        if np.any(n < 0) or np.any(~np.isfinite(n)):
            raise ValueError("photon numbers must be finite and >= 0")
        if e.size > 1 and np.any(np.diff(e) <= 0):
            order = np.argsort(e)
            e, n = e[order], n[order]
            if np.any(np.diff(e) == 0):
                raise ValueError("duplicate energies in spectrum")
        object.__setattr__(self, "energy_ev", e)
        object.__setattr__(self, "photons_per_s", n)

    # -- constructors -------------------------------------------------------------
    @classmethod
    def monochromatic(cls, energy_ev: float, flux: float, label: str = "") -> "Spectrum":
        """A single line of ``flux`` photons/s at ``energy_ev``."""
        return cls(np.array([float(energy_ev)]), np.array([float(flux)]),
                   label or f"{energy_ev / 1e3:g} keV")

    @classmethod
    def lines(cls, lines: Iterable[Tuple[float, float]], label: str = "") -> "Spectrum":
        """Several lines, e.g. a fundamental with its harmonics:
        ``Spectrum.lines([(27e3, 2.4e10), (81e3, 1e7)])``."""
        e, n = zip(*lines)
        return cls(np.array(e, dtype=float), np.array(n, dtype=float), label)

    @classmethod
    def from_density(cls, energy_ev: Sequence[float], photons_per_s_per_ev: Sequence[float],
                     label: str = "") -> "Spectrum":
        """From a spectral density dN/dE in photons/s/eV sampled on ``energy_ev``."""
        e = np.asarray(energy_ev, dtype=float)
        d = np.asarray(photons_per_s_per_ev, dtype=float)
        order = np.argsort(e)
        e, d = e[order], d[order]
        return cls(e, d * _trapezoid_weights(e), label)

    @classmethod
    def from_per_bandwidth(cls, energy_ev: Sequence[float], photons_per_s_per_01bw: Sequence[float],
                           label: str = "") -> "Spectrum":
        """From a flux per 0.1 % bandwidth (the usual synchrotron unit).

        ``dN/dE = N_0.1%BW / (1e-3 E)``: a fixed relative bandwidth spans an
        energy interval proportional to E.
        """
        e = np.asarray(energy_ev, dtype=float)
        return cls.from_density(e, np.asarray(photons_per_s_per_01bw, dtype=float) / (1e-3 * e), label)

    @classmethod
    def from_file(cls, path: str, kind: str = "density", energy_unit: str = "eV",
                  usecols: Tuple[int, int] = (0, 1), label: str = "", **loadtxt_kw) -> "Spectrum":
        """Read a two-column text file.

        ``kind``: ``"density"`` (photons/s/eV), ``"per_kev"`` (photons/s/keV),
        ``"per_0.1%bw"`` or ``"bins"`` (photons/s per line).  ``energy_unit``:
        ``"eV"`` or ``"keV"``.
        """
        e, y = np.loadtxt(path, usecols=usecols, unpack=True, **loadtxt_kw)
        e = e * (1e3 if energy_unit.lower() == "kev" else 1.0)
        kind = kind.lower()
        if kind == "density":
            return cls.from_density(e, y, label or str(path))
        if kind == "per_kev":
            return cls.from_density(e, y / 1e3, label or str(path))
        if kind in ("per_0.1%bw", "per_01bw", "0.1%bw"):
            return cls.from_per_bandwidth(e, y, label or str(path))
        if kind == "bins":
            return cls(e, y, label or str(path))
        raise ValueError("kind must be 'density', 'per_kev', 'per_0.1%bw' or 'bins'")

    # -- properties ---------------------------------------------------------------
    @property
    def total_flux(self) -> float:
        """Photons/s."""
        return float(self.photons_per_s.sum())

    @property
    def power_w(self) -> float:
        """Beam power, W."""
        return float((self.photons_per_s * self.energy_ev).sum() * J_PER_EV)

    @property
    def mean_energy_ev(self) -> float:
        """Photon-number-weighted mean energy, eV."""
        return float((self.photons_per_s * self.energy_ev).sum() / self.photons_per_s.sum())

    @property
    def is_monochromatic(self) -> bool:
        return self.energy_ev.size == 1

    def density(self) -> np.ndarray:
        """Spectral density, photons/s/eV (bins divided by their trapezoid width)."""
        if self.energy_ev.size == 1:
            raise ValueError("a single line has no density")
        return self.photons_per_s / _trapezoid_weights(self.energy_ev)

    def band(self, frac: float = 0.01) -> Tuple[float, float]:
        """Energies where the density first and last exceeds ``frac`` of its peak."""
        if self.is_monochromatic:
            return float(self.energy_ev[0]), float(self.energy_ev[0])
        d = self.density()
        idx = np.flatnonzero(d >= frac * d.max())
        return float(self.energy_ev[idx[0]]), float(self.energy_ev[idx[-1]])

    # -- operations ---------------------------------------------------------------
    def scaled_to(self, total_flux: float) -> "Spectrum":
        """Same shape, total flux set to ``total_flux`` photons/s (e.g. a diode reading)."""
        return replace(self, photons_per_s=self.photons_per_s * (total_flux / self.total_flux))

    def scaled(self, factor: float) -> "Spectrum":
        return replace(self, photons_per_s=self.photons_per_s * factor)

    def attenuated(self, transmission: np.ndarray, label: Optional[str] = None) -> "Spectrum":
        """Multiply each line by a transmission (array of the same length)."""
        t = np.broadcast_to(np.asarray(transmission, dtype=float), self.energy_ev.shape)
        return replace(self, photons_per_s=self.photons_per_s * t, label=label or self.label)

    def through(self, material, thickness_um: float) -> "Spectrum":
        """Spectrum after a slab (filter, window, air path) of ``material``."""
        return self.attenuated(material.transmission(self.energy_ev, thickness_um))

    def restricted(self, e_min_ev: float = 0.0, e_max_ev: float = math.inf) -> "Spectrum":
        """Keep only the lines within [e_min, e_max]."""
        m = (self.energy_ev >= e_min_ev) & (self.energy_ev <= e_max_ev)
        if not np.any(m):
            raise ValueError("no line left in the requested range")
        return replace(self, energy_ev=self.energy_ev[m], photons_per_s=self.photons_per_s[m])


class Beam:
    """X-ray beam at the sample: spectrum, transverse size and profile.

    Parameters
    ----------
    spectrum : Spectrum
    width_um, height_um : float
        Horizontal and vertical size (full size for ``"tophat"``, FWHM for
        ``"gaussian"``), um, measured perpendicular to the beam.
    profile : {"tophat", "gaussian"}

    Legacy (v0) keywords ``beamEnergy`` (eV), ``photonFlux`` (ph/s),
    ``beamWidth``, ``beamHeight`` (um) and ``exposure`` (s) are still accepted
    and build a monochromatic top-hat beam.

    Examples
    --------
    >>> Beam.monochromatic(27e3, 2.44e10, width_um=630, height_um=110)
    >>> Beam(Spectrum.from_density(e, dnde), width_um=0.5, height_um=0.5, profile="gaussian")
    """

    def __init__(self, spectrum: Optional[Spectrum] = None, width_um: Optional[float] = None,
                 height_um: Optional[float] = None, profile: str = "tophat", *,
                 beamEnergy=None, photonFlux=None, beamWidth=None, beamHeight=None, exposure=None):
        legacy = any(v is not None for v in (beamEnergy, photonFlux, beamWidth, beamHeight, exposure))
        if legacy:
            if spectrum is not None:
                raise TypeError("do not mix the spectrum argument with the v0 keywords")
            if None in (beamEnergy, photonFlux, beamWidth, beamHeight):
                raise TypeError("v0 keywords: beamEnergy, photonFlux, beamWidth and beamHeight are required")
            spectrum = Spectrum.monochromatic(beamEnergy, photonFlux)
            width_um, height_um = beamWidth, beamHeight
        if spectrum is None or width_um is None or height_um is None:
            raise TypeError("Beam needs a spectrum, width_um and height_um")
        if width_um <= 0 or height_um <= 0:
            raise ValueError("beam sizes must be positive")
        profile = profile.lower()
        if profile not in ("tophat", "gaussian"):
            raise ValueError("profile must be 'tophat' or 'gaussian'")
        self.spectrum = spectrum
        self.width_um = float(width_um)
        self.height_um = float(height_um)
        self.profile = profile
        #: exposure carried over from the v0 API (seconds), or None
        self.exposure_s = None if exposure is None else float(exposure)

    @classmethod
    def monochromatic(cls, energy_ev: float, flux: float, width_um: float, height_um: float,
                      profile: str = "tophat") -> "Beam":
        return cls(Spectrum.monochromatic(energy_ev, flux), width_um, height_um, profile)

    # -- fluence factors -----------------------------------------------------------
    @property
    def nominal_area_cm2(self) -> float:
        """``width x height`` in cm^2."""
        return self.width_um * self.height_um * CM2_PER_UM2

    @property
    def peak_factor(self) -> float:
        """Peak fluence in units of ``N / (w h)``."""
        return 1.0 if self.profile == "tophat" else 4.0 * math.log(2.0) / math.pi

    @property
    def mean_factor(self) -> float:
        """Signal-weighted mean fluence in units of ``N / (w h)``."""
        return 1.0 if self.profile == "tophat" else 2.0 * math.log(2.0) / math.pi

    @property
    def sigma_um(self) -> Tuple[float, float]:
        """Gaussian standard deviations (horizontal, vertical); tophat: (w/sqrt12, h/sqrt12)."""
        if self.profile == "gaussian":
            k = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            return self.width_um * k, self.height_um * k
        return self.width_um / math.sqrt(12), self.height_um / math.sqrt(12)

    # v0 attribute names --------------------------------------------------------
    @property
    def beamEnergy(self):  # noqa: N802
        if not self.spectrum.is_monochromatic:
            raise AttributeError("beamEnergy is only defined for a monochromatic beam")
        return float(self.spectrum.energy_ev[0])

    @property
    def photonFlux(self):  # noqa: N802
        return self.spectrum.total_flux

    @property
    def beamWidth(self):  # noqa: N802
        return self.width_um

    @property
    def beamHeight(self):  # noqa: N802
        return self.height_um

    @property
    def exposure(self):
        return self.exposure_s

    def __repr__(self) -> str:
        s = self.spectrum
        en = (f"{s.energy_ev[0] / 1e3:g} keV" if s.is_monochromatic
              else f"{s.energy_ev[0] / 1e3:g}-{s.energy_ev[-1] / 1e3:g} keV ({s.energy_ev.size} pts)")
        return (f"Beam({en}, {s.total_flux:.3g} ph/s, {self.width_um:g} x {self.height_um:g} um, "
                f"{self.profile})")
