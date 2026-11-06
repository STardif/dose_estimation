"""The v0 (2022-2023) API, kept so that the notebook of the paper still runs.

``Beam``, ``Material`` and ``Layer`` accept their v0 keywords directly (see
their docstrings); this module adds :class:`Stacking`, whose ``get_dose``
writes the v0 result attributes onto the layers and materials.

Physics changed in v1: the deposited energy now uses the mass
energy-absorption coefficient.  ``Stacking.get_dose(beam,
deposition="attenuation")`` gives the v0 model (with NIST attenuation
coefficients instead of Henke + Klein-Nishina, i.e. 3-10 % higher at 27 keV).
"""

from __future__ import annotations

import warnings

from .beam import Beam
from .stack import Layer, Stack

__all__ = ["Stacking"]

_WARNED = False


class Stacking(Stack):
    """v0 name of :class:`Stack`; ``get_dose`` mimics the v0 behaviour.

    After ``get_dose(beam)``, each material carries ``absorption`` (fraction
    of the photons entering its layer that interact in it), ``absorbedEnergy``
    (J), ``probedVolume`` (cm^3), ``probedWeigth`` (g), ``dose_kGy`` and
    ``doserate_kGys``; each layer carries ``probedVolume``,
    ``incidentPhotonFlux`` and ``transmittedPhotonFlux``.  The full result is
    in ``self.result``.
    """

    def __init__(self, layerList, incidence_angle_deg: float = 0.0):  # noqa: N803
        if len(set(map(id, layerList))) != len(layerList):
            raise ValueError("You can not use the same Layer object more than once in layerList")
        mats = [m for L in layerList for m, _ in L.components]
        if len(set(map(id, mats))) != len(mats):
            # v1 can reuse materials, but the v0 attributes written on them would be ambiguous
            warnings.warn("a Material is used in several layers: its v0 result attributes "
                          "(dose_kGy, ...) will hold the values of the last one; use "
                          "Stacking.result or Stack.compute instead.", stacklevel=2)
        super().__init__(layerList, incidence_angle_deg=incidence_angle_deg)
        self.result = None

    def get_dose(self, xRays: Beam, deposition: str = "energy_absorption", **kw):  # noqa: N803
        global _WARNED
        if not _WARNED and deposition == "energy_absorption":
            warnings.warn("dose_estimation v1: the deposited energy now uses mu_en/rho "
                          "(NIST). Doses of light materials are 2-3x lower than with v0; "
                          "pass deposition='attenuation' to reproduce the 2023 values.",
                          FutureWarning, stacklevel=2)
            _WARNED = True
        exposure = xRays.exposure_s if xRays.exposure_s is not None else kw.pop("exposure_s", None)
        if exposure is None:
            raise ValueError("no exposure: give Beam(..., exposure=...) or exposure_s=...")
        res = self.compute(xRays, exposure_s=exposure, deposition=deposition, **kw)
        self.result = res
        area = xRays.nominal_area_cm2
        for L, LR in zip(self.layers, res.layers):
            L.probedVolume = area * L.thickness_um / 1e4
            L.incidentPhotonFlux = LR.incident_flux
            L.transmittedPhotonFlux = LR.transmitted_flux
            for (m, v), c in zip(L.components, LR.components):
                m.probedVolume = L.probedVolume * v
                m.probedWeigth = c.mass_g
                m.absorption = c.interaction_fraction_layer
                m.absorbedEnergy = c.absorbed_power_w * exposure
                m.dose_kGy = c.dose / 1e3
                m.doserate_kGys = c.dose_rate / 1e3
        return res

    def get_visual_dose(self, show: bool = True):
        """v0 figure of the stack, plus the printed summary."""
        if self.result is None:
            raise RuntimeError("call get_dose first")
        import matplotlib.pyplot as plt
        from .plotting import plot_stack
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_stack(self.result, ax=ax)
        print(self.result.summary())
        if show:
            plt.show()
        return fig, ax
