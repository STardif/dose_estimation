"""Cross-check of the photoelectron-escape estimate against the electron Monte Carlo.

Reproduces the tables quoted in README.md (section "Validation"):

1. backscattering coefficients of the Monte Carlo against the Hunger-Kuechler fit of measurements;
2. the micro-beam case of Sanishvili et al. (PNAS 2011): protein crystal, 18.5 keV, 0.84 vs 15.6 um;
3. the BM32 micro-Laue case: 0.5 um Gaussian beam on Si, Ge, NMC811 and an organic electrolyte,
   analytic estimate (v1.0 constant detour 0.4, and the Z-dependent default) versus Monte Carlo
   (infinite medium with isotropic emission; Sauter emission; free surface at 40 deg grazing).

Run from the repository root:  python validation/escape_crosscheck.py   (about 1-2 minutes)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dose_estimation import Material, electron_mc as emc, escape  # noqa: E402


def main():
    print("1. Backscattering coefficient at 20 keV, normal incidence")
    for sym, rho in [("C", 2.0), ("Al", 2.7), ("Cu", 8.96), ("Au", 19.3)]:
        med = emc.ElectronMedium({sym: 1.0}, rho)
        print(f"   {sym:2s}  MC {emc.backscatter_coefficient(med, 20.0, n=3000):.3f}   "
              f"Hunger-Kuechler {emc.hunger_kuchler(med.z[0], 20.0):.3f}")

    print("\n2. Sanishvili et al. 2011: protein crystal at 18.5 keV, signal-weighted retention")
    prot = {"C": 0.30, "H": 0.08, "N": 0.08, "O": 0.53, "S": 0.01}
    res = {}
    for d in (0.84, 15.6):
        a04 = escape.component_retention(prot, prot, 1.2, [18.5e3], d, d, "gaussian", 0.4)[0][0]
        az = escape.component_retention(prot, prot, 1.2, [18.5e3], d, d, "gaussian")[0][0]
        mc = emc.retention_mc(prot, 1.2, 18.5e3, d, d, "gaussian", None, "sauter", n_electrons=3000).mean
        res[d] = (a04, az, mc)
        print(f"   beam {d:5.2f} um: analytic(0.4) {a04:.3f}  analytic(Z) {az:.3f}  MC {mc:.3f}")
    print("   dose ratio 15.6/0.84 um: " + ", ".join(
        f"{n} {res[15.6][i] / res[0.84][i]:.1f}x" for i, n in enumerate(["analytic(0.4)", "analytic(Z)", "MC"]))
        + "   (damage ratio measured: ~3x)")

    print("\n3. BM32 micro-Laue: 0.5 um FWHM Gaussian, signal-weighted retention")
    mats = {"Si": Material("Si", 2.33), "Ge": Material("Ge", 5.32),
            "NMC811": Material("LiNi0.8Mn0.1Co0.1O2", 4.8), "electrolyte": Material("C3H6O3", 1.15)}
    print(f"   {'material':12s}{'keV':>5s}{'an. 0.4':>9s}{'an. Z':>8s}{'MC iso':>8s}{'MC Sauter':>11s}{'MC surface':>12s}")
    for name, m in mats.items():
        for e in (10e3, 15e3, 20e3, 25e3):
            a04 = escape.component_retention(m.mass_fractions, m.mass_fractions, m.density, [e], 0.5, 0.5,
                                             "gaussian", 0.4)[0][0]
            az = escape.component_retention(m.mass_fractions, m.mass_fractions, m.density, [e], 0.5, 0.5,
                                            "gaussian")[0][0]
            mi = emc.retention_mc(m.mass_fractions, m.density, e, 0.5, 0.5, "gaussian", None, "isotropic",
                                  n_electrons=2000).mean
            ms = emc.retention_mc(m.mass_fractions, m.density, e, 0.5, 0.5, "gaussian", None, "sauter",
                                  n_electrons=2000).mean
            mf = emc.retention_mc(m.mass_fractions, m.density, e, 0.5, 0.5, "gaussian", 40.0, "sauter",
                                  n_electrons=2000).mean
            print(f"   {name:12s}{e / 1e3:5.0f}{a04:9.3f}{az:8.3f}{mi:8.3f}{ms:11.3f}{mf:12.3f}")


if __name__ == "__main__":
    main()
