# dose_estimation

X-ray dose and dose rate in layered samples at synchrotrons — battery cells measured operando, thin films, bulk crystals — for monochromatic, pink or white beams.

Companion code of T. Jousseaume, J.-F. Colin, M. Chandesris, S. Lyonnard, S. Tardif, *How beam damage can skew synchrotron operando studies of batteries*, ACS Energy Lett. **8**, 3323 (2023), [doi:10.1021/acsenergylett.3c00815](https://doi.org/10.1021/acsenergylett.3c00815).

Version 1 changes the physics of the deposited energy (see [What changed in v1](#what-changed-in-v1)).

## Install

```bash
git clone https://github.com/STardif/dose_estimation
cd dose_estimation
pip install -e ".[plot,tables]"          # numpy + scipy are the only hard dependencies
pytest
```

All X-ray data (NIST tables for Z = 1–92) ship with the package; nothing is downloaded.

## Quick start

```python
from dose_estimation import Material, Layer, Stack, Beam

nmc = Material("LiNi0.8Mn0.1Co0.1O2", 4.8, name="NMC811")
ely = Material("C3H6O3", 1.15, name="electrolyte")          # one instance for every layer
cell = Stack([
    Layer.solid("Cu collector", Material("Cu", 8.96), thickness_um=10),
    Layer.porous("anode", Material("C", 2.3, name="graphite"), porosity=0.40, filler=ely, thickness_um=54),
    Layer.porous("cathode", nmc, porosity=0.28, filler=ely, thickness_um=32),
])
beam = Beam.monochromatic(27e3, 2.44e10, width_um=630, height_um=110)   # eV, photons/s, um
res = cell.compute(beam, exposure_s=10631)

print(res.summary("kGy"))
res.table("kGy")                          # pandas DataFrame, one row per component
res.get("cathode", "NMC811").dose         # Gy
```

Units are in the argument names: energies in eV, sizes and thicknesses in µm, densities in g/cm³, flux in photons/s, rates in Gy/s.

## What is computed

For each photon energy *E* of the spectrum and each layer *j* (upstream first), with the beam at an angle θ from the layer normal (path *L* = *t* / cos θ):

* **Attenuation** uses the mass attenuation coefficient μ/ρ (photo-absorption + coherent + incoherent scattering). The layer's effective coefficient is μ\* = Σ_c v_c ρ_c (μ/ρ)_c over its components (volume fractions v_c), and the flux entering layer *j*+1 is N_j(E) exp(−μ\* L). Beam hardening follows automatically.
* **Deposited energy** uses the mass energy-absorption coefficient μ_en/ρ, which excludes the energy carried away by scattered and fluorescence photons. The depth-averaged dose rate of component *c* is

  D_c = Σ_E  f · N_j(E) / A · E · (μ_en/ρ)_c · (1 − e^(−x)) / x,  x = μ\* L

  with *A* = width × height and *f* the beam-profile factor. This is the derivation of SI-6 of the paper with μ_en in place of μ for the deposited energy (ASTM E666; Hubbell & Seltzer, NISTIR 5632).

Each component of each layer reports:

| field | meaning |
| --- | --- |
| `dose_rate` | depth-averaged, profile-mean dose rate (Gy/s): the quantity of the 2023 paper |
| `dose_rate_max` | at the entrance face and the beam centre |
| `kerma_rate`, `kerma_rate_max` | the same without photoelectron escape |
| `dose`, `dose_max` | × exposure, when `exposure_s` is given |
| `interaction_fraction_layer` | fraction of the photons entering the layer that interact in this component (the v0 "absorption") |
| `energy_fraction`, `absorbed_power_w` | share of the incident beam power deposited in it |

and each layer its `transmission` and `mixed_dose_rate` (see below). `DoseResult` also offers `depth_profile()`, `spectral_contribution()`, `time_to_dose()`, `with_exposure()`, `table()`, `summary()` and `plot()`.

### Bounds the homogenised model cannot resolve

* **Electron sharing between phases.** Per-component doses assume that each secondary electron stops in the phase that produced it. Photoelectron ranges (1–10 µm at 10–30 keV) are comparable to particle sizes, so electrons from NMC deposit energy in the surrounding electrolyte and binder. `mixed_dose_rate` is the opposite limit, where all phases of a layer share the energy (dose = absorbed power / layer mass). The true dose of a minor phase next to a strong absorber lies between the two; for the 2023 cathode the gap is 25–40×.
* **Fluorescence.** μ_en assumes fluorescence photons leave the volume of interest. For large beams part of them is reabsorbed in the footprint; `deposition="attenuation"` (every interacting photon deposits all its energy locally, the 2023 model) is an upper bound.
* **Photoelectron escape (micro-beams).** `compute(..., electron_escape=True)` estimates the energy carried out of the footprint by photoelectrons and Auger electrons (Kanaya–Okayama ranges, straight tracks, a detour factor of 0.4 calibrated on micro-beam damage experiments). It matters below a few µm and dominates for sub-micron KB foci. `detour_factor=1.0` gives the largest plausible escape; the kerma is the other bound. See `dose_estimation/escape.py` for the model and its validation; use a Monte Carlo code (RADDOSE-3D, PENELOPE, Geant4) for quantitative work.

### Beams

```python
Beam.monochromatic(energy_ev, flux, width_um, height_um, profile="tophat" | "gaussian")
Beam(Spectrum, width_um, height_um, profile)
```

* `"tophat"`: uniform over width × height; mean = peak fluence = N/(wh).
* `"gaussian"`: width and height are FWHM; peak fluence 0.883 N/(wh), signal-weighted mean 0.441 N/(wh) (the dose seen by a signal proportional to the local fluence, as in diffraction). The convention alone changes the reported dose by up to 2.3×.

Spectra: `Spectrum.monochromatic`, `Spectrum.lines` (fundamental + harmonics), `Spectrum.from_density` (photons/s/eV), `Spectrum.from_per_bandwidth` (photons/s/0.1 %BW), `Spectrum.from_file`, and `.through(material, thickness_um)`, `.scaled_to(total_flux)`, `.restricted(...)`.

### Geometry

`Stack(layers, incidence_angle_deg=0)` or `Stack(layers, grazing_angle_deg=40)` (BM32 micro-Laue). The last layer can be semi-infinite (`thickness_um=np.inf`) for a bulk sample in reflection; its `dose_rate` is then averaged over the first attenuation length, and `dose_rate_max` is the surface value.

### Materials

```python
Material("Li(Ni0.8Mn0.1Co0.1)O2", 4.8)                     # parentheses, decimals, hydrates "CuSO4*5H2O"
Material.from_mass_fractions({nmc: 0.96, pvdf: 0.025, carbon: 0.015})
solvent = Material.from_volume_fractions({ec: 1, pc: 1, dmc: 3})
electrolyte = Material.solution(solvent, "LiPF6", molarity=1.0, density=1.27)
Layer("cathode", components=[(nmc, 0.67), (pvdf, 0.05), (carbon, 0.03), (electrolyte, 0.25)], thickness_um=32)
```

### Exposure histories and maps

`ExposureSchedule.periodic(on_s, period_s, n)`, `.continuous(t)`, `.dose_history(rate)`, `.mean_dose_rate(rate)` for intermittent measurements; `raster_factor(beam, dx_um, dy_um)` for the map-averaged dose of a raster scan with overlapping (or sparse) exposures.

## White beam at BM32 (micro-Laue)

The absolute spectrum at the sample is computed by `xrf_beamline` of [pyXRT](https://github.com/STardif/pyXRT) (bending magnet, Ir mirrors, Be windows, slits traced to the source, 3.3 m of incident air). It needs `xraylib`.

```python
from dose_estimation import Beam, Layer, Material, Stack
from dose_estimation.sources import bm32_white_beam, bm32_white_beam_from_h5

spectrum = bm32_white_beam(pyxrt_path="~/pyXRT")                      # default 2026 slit settings
# spectrum = bm32_white_beam_from_h5("scan.h5", scan="1.1", pyxrt_path="~/pyXRT")
beam = Beam(spectrum, width_um=0.5, height_um=0.5, profile="gaussian")   # use the measured KB spot
bulk = Stack([Layer.solid("Ge", Material("Ge", 5.32), float("inf"))], grazing_angle_deg=40)
res = bulk.compute(beam, exposure_s=1.0, electron_escape=True)
```

Any pyXRT `Beamline` works through `spectrum_from_pyxrt(beamline, energies_kev)`. The pyXRT spectrum already includes the incident air path: do not add it as a layer. pyXRT assumes ideal mirrors and nominal slit gaps; rescale with `spectrum.scaled_to(measured_flux)` when a diode or ion-chamber reading is available.

`white_beam_BM32.ipynb` walks through the spectrum, doses for several bulk crystals, the spectral origin of the dose, a battery electrode and raster maps.

## What changed in v1

* **Deposited energy uses μ_en/ρ** (NIST) instead of the attenuation coefficient. At 27 keV the NMC811 and metal-foil doses drop by 11–13 %, the graphite and organic electrolyte doses by 52–65 %; the gap grows with energy for light phases. `deposition="attenuation"` reproduces the 2023 model.
* **Attenuation uses NIST μ/ρ**, which includes Rayleigh scattering and bound-electron Compton scattering (the v0 Henke photo-absorption + free-electron Klein–Nishina total was 3–10 % low at 27 keV and stopped at 30 keV).
* Spectra, Gaussian beams, incidence angle, semi-infinite layers, depth profiles, N-component layers, mixtures and solutions, mixed-layer bound, photoelectron-escape estimate, exposure schedules, white beam from pyXRT.
* Results are returned (`DoseResult`), not written onto the inputs; materials can be reused across layers.
* Packaging: installable package, bundled data, no download at import, tests. Fixed: the v0 auto-download (`urllib.request` was never imported), the iodine atomic weight in `molw.dat`, an invalid regex escape, three NIST files with rows out of order.

The v0 API still runs (`Beam(beamEnergy=...)`, `Layer(solidMaterial=..., thickness=<mm>)`, `Stacking(...).get_dose()`, `get_visual_dose()`); it warns once that the default physics changed. See `CHANGELOG.md`.

## Validation

* NIST table values are reproduced exactly at the tabulated energies, edges included; between them the interpolation is a monotone cubic in log–log space, piecewise between edges, within 6 % of the Chantler totals of XrayDB over 7–75 keV (Chantler's total omits part of the coherent term at low energy).
* Thin-sample limit: D = Φ E μ_en/ρ exactly; energy conservation in `deposition="attenuation"` (absorbed + transmitted = incident power) to 1e-10; a 10 eV-wide white spectrum reproduces the monochromatic dose to 1e-5.
* Electron range: Kanaya–Okayama gives 8.9 × 10⁻⁴ g/cm² in water at 20 keV (ESTAR CSDA: 8.6 × 10⁻⁴).
* Escape estimate against micro-beam experiments on protein crystals at 18.5 keV: 4.2× less dose per calculated dose from a 15.6 µm to a 0.84 µm beam (measured 3×, Sanishvili et al., PNAS 2011) and 2.5× for a 0.7 µm line focus (measured 2.0–2.5×, Finfrock et al., Acta Cryst. D 2010).
* The v0 notebook outputs are reproduced within 1.5 % by the v0 code run with Chantler f2 tables, and within 3–10 % by `deposition="attenuation"` (the Rayleigh term).

## Limits

The model is one-dimensional and homogenised: no lateral structure, no particle-scale geometry, no electron transport across layer interfaces, no fluorescence reabsorption. Inputs (flux, beam size, porosity, loading) usually carry 10–20 % uncertainties each. The dose thresholds of the 2023 paper (80 kGy, 1–4 MGy, 11 MGy) refer to the v0 model with top-hat beams; the NMC values move by about −11 % with μ_en.

## References

* Hubbell & Seltzer, *Tables of X-Ray Mass Attenuation Coefficients and Mass Energy-Absorption Coefficients*, NISTIR 5632 (1995/2004), https://physics.nist.gov/PhysRefData/XrayMassCoef/
* ASTM E666, *Standard Practice for Calculating Absorbed Dose From Gamma or X Radiation*.
* Christensen et al., *Beam damage in operando X-ray diffraction studies of Li-ion batteries*, J. Synchrotron Rad. 30, 561 (2023).
* Black et al., *Beam effects in synchrotron radiation operando characterization of battery materials*, Chem. Mater. 36, 5596 (2024).
* Bury et al., *Estimate your dose: RADDOSE-3D*, Protein Sci. 27, 217 (2018); Dickerson & Garman, Protein Sci. 30, 8 (2021).
* Sanishvili et al., PNAS 108, 6127 (2011); Finfrock et al., Acta Cryst. D 66 (2010).
* Kanaya & Okayama, J. Phys. D 5, 43 (1972).

## License

See `LICENSE`.
