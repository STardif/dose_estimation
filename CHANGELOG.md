# Changelog

## 1.1.0 — 2026-11

### New
- Fluorescence reabsorption in the beam footprint (`compute(..., fluorescence=True)`, module `fluorescence`): stack-level Monte Carlo of the K and L fluorescence lines with Elam yields, jump ratios and Coster–Kronig transfers; result `fluorescence_ratio` per component and a `FluorescenceReport`.
- Bundled atomic relaxation data (`data/xray_levels.csv`, `xray_lines.csv`, `coster_kronig.csv`, from XrayDB, CC0) and module `atomic`.
- Electron-transport Monte Carlo (`electron_mc`): screened Rutherford + Joy–Luo, Sauter emission, free surface; `compute(..., electron_escape="mc")`; `validation/escape_crosscheck.py`; RADDOSE-3D inputs and comparison script in `validation/raddose3d/`.
- Temperature-rise estimates (`thermal.temperature_rise`): SI-13 through-thickness model, 3D near field along the beam path, plate far field with a heat sink or surface losses, semi-infinite samples.
- Uncertainty propagation (`propagate`, `propagate_stack`, `Uncertain`) with Spearman sensitivities.

### Changed (numbers)
- Photoelectron escape: the detour factor is now Z-dependent (`1.72 Z̄^-0.357`, fitted to the Monte Carlo) instead of 0.4. Escape-corrected doses drop for light materials (organic electrolyte at 20 keV, 0.5 µm beam: retention 0.14 → 0.08) and change little for Fe–Ge. `detour_factor=0.4` restores v1.0.
- The escape model takes binding energies and fluorescence yields from the Elam tables (edges below 1 keV included).


## 1.0.0 — 2026-11

### Physics (changes the numbers)
- Deposited energy uses the NIST mass energy-absorption coefficient μ_en/ρ. v0 counted the full energy of every interacting photon, including Compton- and Rayleigh-scattered and fluorescence photons that leave the sample. At 27 keV: NMC811 −11 %, Cu/Al −12/−13 %, SiO2 −16 %, graphite −65 %, organic electrolyte −52 %. Use `deposition="attenuation"` to reproduce the 2023 model.
- Attenuation uses NIST μ/ρ (with Rayleigh and bound-electron Compton). v0 used Henke f2 + free-electron Klein–Nishina, 3–10 % low at 27 keV, extrapolated without warning above 30 keV and missing K edges above 30 keV.
- Interpolation: monotone cubic (PCHIP) in log–log, piecewise between absorption edges.

### New
- `Spectrum`: lines, harmonics, densities per eV or per 0.1 % BW, files; dose integrated over the spectrum with beam hardening.
- `dose_estimation.sources`: BM32 white beam from pyXRT (`bm32_white_beam`, `bm32_white_beam_from_h5`, `spectrum_from_pyxrt`).
- Gaussian beams (peak and signal-weighted mean), incidence or grazing angle, semi-infinite layers.
- N-component layers; `Material.from_mass_fractions`, `.from_volume_fractions`, `.solution`; formula parser with parentheses and hydrates.
- `DoseResult` with `table`, `summary`, `depth_profile`, `spectral_contribution`, `time_to_dose`, `with_exposure`, `plot`.
- Mixed-layer dose (electrons shared between phases) as the upper bound for minor phases.
- Optional photoelectron-escape estimate for micro-beams (`electron_escape=True`, `detour_factor`).
- `ExposureSchedule` (intermittent/continuous dose histories) and `raster_factor` (maps).
- Optional `xraylib` backend (`set_backend("xraylib")`), useful below 1 keV.

### Fixed
- Import crashed on a fresh interpreter (`urllib.request` not imported) and only worked from the repository folder; no network access or file writes at import any more.
- Iodine atomic weight `126.904 47` in `molw.dat` made every iodine compound fail.
- Invalid escape sequence in the formula regex (SyntaxWarning on Python ≥ 3.12).
- `Material.absorption()` was overwritten by a float after `get_dose`; `absorption()`/`transmission()` used an attribute that was never set.
- Porosity is validated; a porous layer without filler raises a clear error.
- NIST files for Mo, Bi and Fr had rows out of order around an edge; they are sorted on load.

### Compatibility
- `Beam(beamEnergy=..., photonFlux=..., beamWidth=..., beamHeight=..., exposure=...)`, `Layer(name, solidMaterial, liquidMaterial, thickness_mm, porosity)`, `Stacking(...).get_dose()`, `.get_visual_dose()` and the v0 result attributes still work. `get_dose` warns once that the default physics changed.
- Package layout: `dose_estimation/` with data in `dose_estimation/data/`; `pip install -e .`.
