# -*- coding: utf-8 -*-
# Dose estimation 
# 2022-12-14
# Sam Tardif (samuel.tardif@cea.fr)
# Thibaut Jousseaume (thibaut.jousseaume@cea.fr)

import numpy as np
import os
import re
import urllib
import tarfile
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# classical electron radius
r = 2.81794e-13  # cm

# Avogadro number
Na = 6.0221409e23  # 1/mol

# electron charge
q = 1.60217662e-19  # C

# Z and W for the elements
elementData = pd.read_csv("molw.dat", skiprows=3)

# check if the scattering factors files are present, if not then download them
if not os.path.isdir("sf"):
    print(
        "Could not find the local folder sf with all the scattering factors or the necessary file in it"
    )
    print("Downloading from the web")
    urllib.request.urlretrieve(
        "https://henke.lbl.gov/optical_constants/sf.tar.gz", "sf.tar.gz"
    )
    file = tarfile.open("sf.tar.gz")
    file.extractall("./sf")
    file.close()


class Beam:
    """
    Describe the X-ray beam parameters.

    Keyword arguments:
    beamEnergy: float
        X-ray beam energy, in eV
    photonFlux: float
        X-ray photon flux just before the sample, in photons/s
    beamWidth: float
        first dimension of the X-ray beam, in um
    beamHeight: float
        second dimension of the X-ray beam, in um
    exposure: float
        total exposure, in s
    """
    def __init__(self, beamEnergy, photonFlux, beamWidth, beamHeight, exposure):
        self.beamEnergy = beamEnergy  # eV
        self.photonFlux = photonFlux  # photon/s
        self.beamWidth = beamWidth  # um
        self.beamHeight = beamHeight  # um
        self.photonEnergy = beamEnergy * q  # J/photon
        self.exposure = exposure  # s


class Atom:
    """
    Used to compute the energy-dependant scattering and absorption of a given element.

    Keyword arguments:
    symbol: string
        International symbol of the element, e.g. "H"
    """
    def __init__(self, symbol):
        self.symbol = symbol

        # read the Z and W from the pandas dataframe, file molw.dat
        self.Z = elementData.loc[elementData["symbol"] == self.symbol][
            "atomic_number"
        ].values[0]
        self.molecularWeigth = float(
            elementData.loc[elementData["symbol"] == self.symbol][
                "molecular_weight"
            ].values[0]
        )

        # read the atomic scattering coeff from the .nff file
        self._fenergy, self._f1, self._f2 = np.loadtxt(
            os.path.join("sf", f"{self.symbol}.nff".lower()), skiprows=1, unpack=True
        )

        # define the interpolation function for f2
        # we set f2 = 0 out of the interpolation range, for E>30keV the Compton dominates anyway
        self.f2 = interp1d(
            self._fenergy, self._f2, bounds_error=False, fill_value=(0, 0)
        )
        self.f2log = interp1d(
            np.log10(self._fenergy),
            np.log10(self._f2),
            bounds_error=False,
            fill_value="extrapolate",
        )

    def get_sigmaCoherent(self, energy):
        """
        energy in eV
        """
        wavelength = (12398 / energy) * 1e-8  # cm
        # return 2*r*wavelength*self.f2(energy) #cm2/atom  #photon absorption
        return (
            2 * r * wavelength * (10 ** self.f2log(np.log10(energy)))
        )  # cm2/atom  #photon absorption

    def get_sigmaIncoherent(self, energy):
        """
        energy in eV
        using the Klein-Nishima approximation
        only correct at relatively high energy but not too much
        """
        k = energy / 511000  # energy in units of electron rest energy, with E in eV
        return (
            8
            * np.pi
            * (r**2)
            * (1 + 2 * k + 1.2 * k**2)
            / (3 * (1 + 2 * k) ** 2)
            * self.Z
        )  # cm2/atom # Compton scattering

    def sigma(self, energy):
        """
        energy in eV
        total cross-section/atom (valid up to few 100 keV)
        """
        return self.get_sigmaCoherent(energy) + self.get_sigmaIncoherent(
            energy
        )  # cm2/atom


class Material:
    """
    Made up from different elements from the chemical formula

    Keyword arguments:
    formula: string
        Standard formula, e.g. C2H6
    density: float
        Materials density, in g/cm3
    name: string, optional
        Material name
    """
    def __init__(self, formula, density, name=None):
        self.name = name
        self.formula = formula
        self.density = density
        self.formulaParsed = self.readFormula(formula)
        self.atoms = [
            (Atom(elem), nelem)
            for elem, nelem in zip(self.formulaParsed[::2], self.formulaParsed[1::2])
        ]
        self.totalMolecularWeight = sum(
            (atom.molecularWeigth * nelem for (atom, nelem) in self.atoms)
        )

    def get_sigmaCoherent(self, energy):
        """
        Rayleigh, photon absorption cross-section, in cm2
        """
        return sum(
            (atom.get_sigmaCoherent(energy) * nelem for (atom, nelem) in self.atoms)
        )  # cm2

    def get_sigmaIncoherent(self, energy):
        """
        energy in eV
        Compton, using the Klein-Nishima approximation
        only correct at relatively high energy but not too much (1E4 - 1E5 eV...)
        """
        return sum(
            (atom.get_sigmaIncoherent(energy) * nelem for (atom, nelem) in self.atoms)
        )  # cm2

    def sigma(self, energy):
        """
        total absorption cross-section, in cm2
        """
        return self.get_sigmaCoherent(energy) + self.get_sigmaIncoherent(energy)  # cm2

    def massAbsCoeff(self, energy):
        """
        mass absorption coefficient, in cm2/g
        """
        return Na * self.sigma(energy) / self.totalMolecularWeight  # cm2/g

    def linAbsCoeff(self, energy):
        """
        linear absorption coefficient, in 1/cm
        """
        return self.density * self.massAbsCoeff(energy)  # 1/cm

    def attenLength(self, energy):
        """
        attenuation length, in um
        """
        return 1e4 / self.linAbsCoeff(energy)  # um

    def readFormula(self, formula):
        """
        parser to read the compositon formula in the form, e.g. LiNi0.2Mn0.4Co.4
        returns a list of successive element symbol (string) and multiplicity (float)
        """
        try:
            fParsed = re.findall(
                r"[A-Z][a-z]*|\d*\.?\d*",
                re.sub("[A-Z][a-z]*(?![\d\.a-z])", r"\g<0>1", formula),
            )[:-1]
            fParsed[1::2] = map(np.float64, fParsed[1::2])
        except:
            raise ValueError("Error reading formula")
        else:
            return fParsed

    def absorption(self, beamEnergy):
        """
        if the Materials.length is defined, compute the absorption fraction at a given energy 
        """
        if hasattr(self, "length"):
            return 1 - np.exp(-self.length / (self.attenLength(beamEnergy)))

    def transmission(self, beamEnergy):
        """
        if the Materials.length is defined, compute the transmission fraction at a given energy 
        """
        if hasattr(self, "length"):
            return np.exp(-self.length / (self.attenLength(beamEnergy)))

    def material_info_label(self, label=""):
        """
        build a summary of the absorption, if computed 
        """
        try:
            label += f"  |_{self.name}\n"
            label += f"      |_density = {self.density:.3f} g/cm3\n"
            label += f"      |_probed volume = {self.probedVolume:.2e} cm3\n"
            label += f"      |_absorption = {self.absorption*100:.2f}%\n"
            label += f"      |_dose = {self.dose_kGy:.3f} kGy\n"
            label += f"      |_doserate = {self.doserate_kGys:.3f} kGy/s\n"
        except Exception as e:
            print(e)
            return "failed to get the material absorption info"
        else:
            return label


class Layer:
    """
    Made up of up to 2 Materials instances, a solidMaterial with a given porosity
    filled with a liquidMaterial. If the porosity is 0, the liquidMaterial is set to None,
    conversely, if the porosity is 1, the solidMaterial is set to None

    Keyword arguments:
    name: string
        To identify the Layer
    solidMaterial: Material
        Material instance
    liquidMaterial: Material
        Material instance
    thickness: float
        Total thickness of the Layer, in mm
    porosity: float
        Porosity of the solidMaterial, between 0 and 1
    """
    def __init__(self, name, solidMaterial, liquidMaterial, thickness, porosity):
        self.name = name
        self.thickness = thickness
        self.porosity = porosity
        if self.porosity == 0.0:
            self.solidMaterial = solidMaterial
            self.liquidMaterial = None
        elif self.porosity == 1.0:
            self.solidMaterial = None
            self.liquidMaterial = liquidMaterial
        else:
            self.solidMaterial = solidMaterial
            self.liquidMaterial = liquidMaterial

    def get_transmission(self, xRays):
        """
        Compute the transmission of the Layer, and thus the absorption and dose.


        Keyword arguments:
        name -- string to identify the Layer
        solidMaterial -- Material instance
        liquidMaterial -- Material instanc
        thickness -- total thickness of the Layer, in mm
        porosity -- porosity of the solidMaterial, between 0 and 1
        """
        porosity = self.porosity
        thickness = self.thickness
        self.probedVolume = (
            thickness / 10 * xRays.beamWidth / 10000 * xRays.beamHeight / 10000
        )  # cm3
        self.incidentPhotonFlux = xRays.photonFlux

        if porosity == 1:
            mu_l = mu_ef = self.liquidMaterial.linAbsCoeff(xRays.beamEnergy)
        elif porosity == 0:
            mu_s = mu_ef = self.solidMaterial.linAbsCoeff(xRays.beamEnergy)
        else:
            mu_s = self.solidMaterial.linAbsCoeff(xRays.beamEnergy)
            mu_l = self.liquidMaterial.linAbsCoeff(xRays.beamEnergy)
            mu_ef = mu_s * (1 - porosity) + mu_l * porosity  # 1/cm

        if porosity < 1:
            sm = self.solidMaterial
            sm.probedVolume = self.probedVolume * (1 - porosity)  # cm3
            sm.probedWeigth = sm.probedVolume * sm.density  # g
            sm.absorption = (
                mu_s / mu_ef * (1 - porosity) * (1 - np.exp(-mu_ef * thickness / 10))
            )  # unitless
            sm.absorbedEnergy = (
                xRays.photonFlux * xRays.photonEnergy * xRays.exposure * sm.absorption
            )  # J
            sm.dose_kGy = sm.absorbedEnergy / (sm.probedWeigth / 1000) / 1000  # kGy
            sm.doserate_kGys = sm.dose_kGy / xRays.exposure  # kGy/s
        if porosity > 0:
            # porosity > 0 means the liquid material is defined
            lm = self.liquidMaterial
            lm.probedVolume = self.probedVolume * porosity  # cm3
            lm.probedWeigth = lm.probedVolume * lm.density  # g
            lm.absorption = (
                mu_l / mu_ef * porosity * (1 - np.exp(-mu_ef * thickness / 10))
            )  # unitless
            lm.absorbedEnergy = (
                xRays.photonFlux * xRays.photonEnergy * xRays.exposure * lm.absorption
            )  # J
            lm.dose_kGy = lm.absorbedEnergy / (lm.probedWeigth / 1000) / 1000  # kGy
            lm.doserate_kGys = lm.dose_kGy / xRays.exposure  # kGy/s

        self.transmittedPhotonFlux = xRays.photonFlux * np.exp(
            -mu_ef * thickness / 10
        )  # photon/s

        return self.transmittedPhotonFlux


class Stacking:
    """
    Describe stacking of the eletrochemical cell using a list of Layers,
    the first Layer in the list is upstream.
    NB: each Layer can only be used once in the list. If necessary, instantiate 
    several similar Layers. Equally, all Layers should be made of different Materials instances

    Keyword arguments:
    layerList -- list of Layers
    """
    def __init__(self, layerList):
        if len(set(layerList)) != len(layerList):
            raise ValueError(
                "You can not use the same Layer object more than once in layerList"
            )
        else:
            # build the list of all materials objects (except None) in all layers
            self.materialsList = [
                l.solidMaterial for l in layerList if l.solidMaterial != None
            ] + [l.liquidMaterial for l in layerList if l.liquidMaterial != None]
            if len(set(self.materialsList)) != len(self.materialsList):
                raise ValueError(
                    "You can not use the same Materials object more than once in all the layers"
                )
            else:
                self.layers = layerList

    def get_dose(self, xRays):
        """
        Given a Beam object, recursively compute the absorption and dose in each layer

        Keyword arguments:
        xRays -- Beam instance
        """
        # copy the incident X-ray beam object to propagate it
        XRays_at_layer = Beam(xRays.beamEnergy, xRays.photonFlux, xRays.beamWidth, xRays.beamHeight, xRays.exposure)
        for layer in self.layers:
            transmittedPhotonFlux = layer.get_transmission(xRays=XRays_at_layer)
            XRays_at_layer = Beam(xRays.beamEnergy, transmittedPhotonFlux, xRays.beamWidth, xRays.beamHeight, xRays.exposure)


    def get_visual_dose(self):
        """
        Make a self-explanatory graph of the results of the absorption and dose computation
        """
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        fig, ax = plt.subplots(figsize=(8, 6))
        t0 = 0
        full_string = ""
        for iii, l in enumerate(self.layers):
            ax.add_patch(
                Rectangle(
                    (t0, 0),
                    l.thickness,
                    1,
                    facecolor=colors[iii],
                    alpha=(1 - l.porosity),
                )
            )
            label = f"-{l.name}\n"
            label += f"  |_thickness = {l.thickness:.3f} mm\n"
            label += f"  |_probed volume = {l.probedVolume:.2e} cm3\n"
            if l.porosity == 0:
                mList = [
                    l.solidMaterial,
                ]
            elif l.porosity == 1:
                mList = [l.liquidMaterial]
            else:
                mList = [l.solidMaterial, l.liquidMaterial]
            for m in mList:
                label = m.material_info_label(label)
            full_string += label
            ax.text(
                t0,
                0.75 + 0.25 * (-1) ** iii,
                label,
                rotation=0,
                va="top",
                fontsize="small",
            )
            t0 += l.thickness

        ax.set_xlim(0, t0)
        ax.set_yticks([])
        ax.set_xlabel("depth along the beam (mm)")
        plt.show()
        print(full_string)
        return fig, ax
