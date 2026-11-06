"""Exposure schedules: dose and dose rate versus time.

Damage depends on the dose *and* on how it is delivered (Jousseaume et al.
2023, experiment 3): a spot measured intermittently receives the same
on-beam dose rate as a continuously exposed one, but a lower time-averaged
rate and recovery periods in between.  :class:`ExposureSchedule` describes
when the beam is on and turns a dose rate into a dose history.

Example
-------
>>> sched = ExposureSchedule.periodic(on_s=1050, period_s=1050 * 3, n=27)   # one spot of three
>>> t, dose = sched.dose_history(result.get("cathode", "NMC").dose_rate)
>>> sched.total_exposure_s, sched.duty_cycle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["ExposureSchedule", "raster_factor"]


@dataclass(frozen=True)
class ExposureSchedule:
    """Beam-on intervals ``[(start_s, duration_s), ...]`` (non-overlapping)."""

    intervals: Tuple[Tuple[float, float], ...]

    def __post_init__(self):
        iv = sorted((float(a), float(d)) for a, d in self.intervals)
        for (a, d) in iv:
            if d < 0:
                raise ValueError("durations must be >= 0")
        for (a1, d1), (a2, _) in zip(iv[:-1], iv[1:]):
            if a2 < a1 + d1 - 1e-12:
                raise ValueError("exposure intervals overlap")
        object.__setattr__(self, "intervals", tuple(iv))

    @classmethod
    def continuous(cls, duration_s: float, start_s: float = 0.0) -> "ExposureSchedule":
        return cls(((start_s, duration_s),))

    @classmethod
    def periodic(cls, on_s: float, period_s: float, n: int, start_s: float = 0.0) -> "ExposureSchedule":
        """``n`` exposures of ``on_s`` seconds every ``period_s`` seconds."""
        if period_s < on_s:
            raise ValueError("period shorter than the exposure")
        return cls(tuple((start_s + k * period_s, on_s) for k in range(int(n))))

    @classmethod
    def from_times(cls, starts_s: Sequence[float], durations_s) -> "ExposureSchedule":
        d = np.broadcast_to(np.asarray(durations_s, dtype=float), np.shape(starts_s))
        return cls(tuple(zip(map(float, starts_s), map(float, d))))

    @property
    def total_exposure_s(self) -> float:
        return float(sum(d for _, d in self.intervals))

    @property
    def span_s(self) -> float:
        """From the first beam-on to the last beam-off."""
        a0 = self.intervals[0][0]
        a1, d1 = self.intervals[-1]
        return a1 + d1 - a0

    @property
    def duty_cycle(self) -> float:
        """Beam-on fraction of the span."""
        return self.total_exposure_s / self.span_s if self.span_s > 0 else 1.0

    def exposure_until(self, t) -> np.ndarray:
        """Accumulated beam-on time at time(s) ``t``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros_like(t)
        for a, d in self.intervals:
            out += np.clip(t - a, 0.0, d)
        return out

    def dose_history(self, dose_rate_gy_s: float, n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Times (s) and cumulative dose (Gy) for a constant on-beam dose rate.

        The time grid includes every interval edge, so the steps are exact.
        """
        edges: List[float] = []
        for a, d in self.intervals:
            edges += [a, a + d]
        t = np.unique(np.concatenate([np.linspace(self.intervals[0][0], edges[-1], n), edges]))
        return t, dose_rate_gy_s * self.exposure_until(t)

    def dose_rate_history(self, dose_rate_gy_s: float, n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Instantaneous dose rate (Gy/s): ``dose_rate`` when the beam is on, else 0."""
        t, _ = self.dose_history(dose_rate_gy_s, n)
        on = np.zeros_like(t, dtype=bool)
        for a, d in self.intervals:
            on |= (t >= a) & (t < a + d)
        return t, np.where(on, dose_rate_gy_s, 0.0)

    def mean_dose_rate(self, dose_rate_gy_s: float) -> float:
        """Time-averaged dose rate over the span (Gy/s)."""
        return dose_rate_gy_s * self.duty_cycle


def raster_factor(beam, step_h_um: float, step_v_um: float) -> float:
    """Map-averaged dose per point, in units of the single-exposure dose rate x dwell time.

    For a raster map with steps ``step_h_um`` x ``step_v_um`` and a dwell
    time ``t`` per point, the dose averaged over the mapped area once the map
    is complete is ``result.dose_rate * t * raster_factor(beam, dx, dy)``
    (``dose_rate`` being the signal-weighted mean of one exposure).  Steps
    smaller than the beam make the exposures overlap (factor > 1); a coarse
    raster spreads the dose (the PCa protocol of the 2023 paper).  The dose a
    point has received *when it is measured* depends on the scan order:
    roughly half of its neighbours have already been exposed.
    """
    if step_h_um <= 0 or step_v_um <= 0:
        raise ValueError("steps must be positive")
    return beam.width_um * beam.height_um / (step_h_um * step_v_um * beam.mean_factor)
