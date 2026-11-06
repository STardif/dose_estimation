"""dose_estimation: X-ray dose and dose rate in layered samples at synchrotrons.

Quick start
-----------
>>> from dose_estimation import Material, Layer, Stack, Beam
>>> nmc = Material("LiNi0.8Mn0.1Co0.1O2", 4.8, name="NMC811")
>>> ely = Material("C3H6O3", 1.15, name="electrolyte")
>>> cell = Stack([Layer.porous("cathode", nmc, porosity=0.28, filler=ely, thickness_um=32)])
>>> res = cell.compute(Beam.monochromatic(27e3, 2.44e10, width_um=630, height_um=110), exposure_s=3600)
>>> print(res.summary())

White beam (BM32 micro-Laue, needs pyXRT + xraylib):

>>> from dose_estimation.sources import bm32_white_beam
>>> beam = Beam(bm32_white_beam(pyxrt_path="~/pyXRT"), width_um=0.5, height_um=0.5, profile="gaussian")

See README.md for the physics and its limits.
"""

from .beam import Beam, Spectrum
from .exposure import ExposureSchedule, raster_factor
from .legacy import Stacking
from .materials import Material, parse_formula
from .stack import ComponentDose, DoseResult, Layer, LayerDose, Stack
from .xray_data import get_backend, set_backend

__version__ = "1.0.0"

__all__ = [
    "Beam", "Spectrum", "Material", "parse_formula", "Layer", "Stack", "Stacking",
    "DoseResult", "ComponentDose", "LayerDose", "ExposureSchedule", "raster_factor",
    "set_backend", "get_backend", "__version__",
]
