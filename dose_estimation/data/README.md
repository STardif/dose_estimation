# Data files

| file | content | source | license |
| --- | --- | --- | --- |
| `NIST_xray_data/*.dat` | mass attenuation μ/ρ and mass energy-absorption μ_en/ρ, Z = 1–92, 1 keV–20 MeV | J. H. Hubbell & S. M. Seltzer, NISTIR 5632 (1995, 2004), https://physics.nist.gov/PhysRefData/XrayMassCoef/ | US government work |
| `molw.dat` | standard atomic weights | IUPAC 2021 | — |
| `xray_levels.csv` | K, L1–L3, M1–M5 edge energies, fluorescence yields, jump ratios | Elam, Ravel & Sieber, Radiat. Phys. Chem. 63, 121 (2002), via XrayDB 9.2 | CC0 (XrayDB) |
| `xray_lines.csv` | emission lines from K and L vacancies: energies, relative intensities | same | CC0 |
| `coster_kronig.csv` | L1→L2, L1→L3, L2→L3 Coster–Kronig probabilities | same | CC0 |

The three CSV files were extracted from `xraydb.sqlite` (https://github.com/xraypy/XrayDB, release 9.2, 2023-12-05), keeping Z ≤ 92.
