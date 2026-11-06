"""Write RADDOSE-3D inputs for an independent check of the photoelectron-escape estimate.

Not run by the test suite (it needs Java and RADDOSE-3D, https://github.com/GarmanGroup/RADDOSE-3D).
Each case is an 8 x 8 x 8 um block hit by a 0.5 um FWHM Gaussian beam, with and without
CALCULATEPEESCAPE.  The ratio of the "Average DWD" (resp. "Max Dose") of the two runs is the
signal-weighted (resp. peak) retention that dose_estimation reports as retention_mean
(resp. retention_peak); compare.py does the comparison.

    python validation/raddose3d/make_inputs.py
    cd validation/raddose3d && for f in *.txt; do java -jar /path/to/raddose3d.jar -i $f -p ${f%.txt}_; done
    python validation/raddose3d/compare.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# integer contents of a cubic cell with the right density (RADDOSE-3D SMALLMOLE needs integers)
CASES = {
    "NMC811": dict(cell=6.955, atoms="Li 10 Ni 8 Mn 1 Co 1 O 20", formula="LiNi0.8Mn0.1Co0.1O2", density=4.8),
    "Ge": dict(cell=5.658, atoms="Ge 8", formula="Ge", density=5.32),
    "Si": dict(cell=5.431, atoms="Si 8", formula="Si", density=2.33),
}
ENERGIES_KEV = (15.0, 20.0)

TEMPLATE = """Crystal
Type Cuboid
Dimensions 8 8 8
PixelsPerMicron 10
AbsCoefCalc SMALLMOLE
UnitCell {cell} {cell} {cell}
NumMonomers 1
SmallMoleAtoms {atoms}
CalculatePEEscape {pe}

Beam
Type Gaussian
Flux 1e10
FWHM 0.5 0.5
Energy {energy}
Collimation Rectangular 2 2

Wedge 0 0
ExposureTime 1
"""


def main():
    for name, c in CASES.items():
        for e in ENERGIES_KEV:
            for pe in ("TRUE", "FALSE"):
                fn = os.path.join(HERE, f"{name}_{e:g}keV_pe{pe.lower()}.txt")
                with open(fn, "w") as fh:
                    fh.write(TEMPLATE.format(cell=c["cell"], atoms=c["atoms"], pe=pe, energy=e))
    print("inputs written to", HERE)


if __name__ == "__main__":
    main()
