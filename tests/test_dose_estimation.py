"""Tests for dose_estimation v1.  Run with ``pytest`` from the repository root."""

import math
import warnings

import numpy as np
import pytest

from dose_estimation import (Beam, ExposureSchedule, Layer, Material, Spectrum, Stack, Stacking,
                             parse_formula)
from dose_estimation import escape, xray_data as xd


# --------------------------------------------------------------------------- data

@pytest.mark.parametrize("sym, e_ev, mu, men", [
    ("O", 10e3, 5.952, 5.565),          # NIST table rows
    ("Cu", 20e3, 33.79, 27.88),
    ("Cu", 8.9799e3, 278.4, 182.4),     # just above the K edge
    ("Cu", 8.9789e3, 38.29, 36.52),     # just below
    ("H", 30e3, 0.3570, 0.01863),
])
def test_nist_table_values(sym, e_ev, mu, men):
    assert xd.mu_rho(sym, e_ev)[0] == pytest.approx(mu, rel=1e-6)
    assert xd.mu_en_rho(sym, e_ev)[0] == pytest.approx(men, rel=1e-6)


def test_interpolation_is_monotone_between_points_and_keeps_edges():
    e = np.linspace(9.0e3, 20e3, 200)            # above the Cu K edge: decreasing
    assert np.all(np.diff(xd.mu_rho("Cu", e)) < 0)
    below = xd.mu_rho("Cu", 8970.0)[0]
    above = xd.mu_rho("Cu", 8990.0)[0]
    assert above / below > 5                     # the K jump survives interpolation


def test_out_of_order_rows_are_fixed():
    # Mo K edge at 19.9995 keV sits next to the 20 keV grid point in the scraped file
    e = np.array([19.99e3, 20.0e3, 20.01e3])
    mu = xd.mu_rho("Mo", e)
    assert mu[0] < 20 and mu[1] > 70 and mu[2] > 70


def test_energy_range_is_enforced():
    with pytest.raises(ValueError):
        xd.mu_rho("C", 500.0)


def test_iodine_weight_parses():
    assert xd.atomic_weight("I") == pytest.approx(126.90447)


def test_fluorescence_yield_reasonable():
    assert xd.fluorescence_yield(28, "K") == pytest.approx(0.41, abs=0.03)    # Ni
    assert xd.fluorescence_yield(29, "K") == pytest.approx(0.44, abs=0.03)    # Cu
    assert xd.fluorescence_yield(6, "K") < 0.01


# ------------------------------------------------------------------------ materials

def test_parse_formula():
    assert dict(parse_formula("Li(Ni0.8Mn0.1Co0.1)O2")) == pytest.approx(
        {"Li": 1, "Ni": 0.8, "Mn": 0.1, "Co": 0.1, "O": 2})
    assert dict(parse_formula("CuSO4*5H2O")) == {"Cu": 1, "S": 1, "O": 9, "H": 10}
    assert dict(parse_formula("LiNi0.2Mn0.4Co.4")) == pytest.approx({"Li": 1, "Ni": .2, "Mn": .4, "Co": .4})
    assert dict(parse_formula("Fe2(SO4)3")) == {"Fe": 2, "S": 3, "O": 12}
    for bad in ["Xx", "Li(Ni", "2", "Li)", ""]:
        with pytest.raises(ValueError):
            parse_formula(bad)


def test_water_mixture_rule():
    w = Material("H2O", 1.0)
    assert w.mass_fractions["H"] == pytest.approx(2 * 1.008 / 18.015, rel=1e-4)
    # NIST water at 20 keV: mu/rho 0.8096, mu_en/rho 0.5503 (element mixture, within 1 %)
    assert w.mu_rho(20e3)[0] == pytest.approx(0.8096, rel=0.01)
    assert w.mu_en_rho(20e3)[0] == pytest.approx(0.5503, rel=0.01)


def test_mixtures():
    ec = Material("C3H4O3", 1.32, "EC")
    dmc = Material("C3H6O3", 1.07, "DMC")
    blend = Material.from_volume_fractions({ec: 1, dmc: 1})
    assert blend.density == pytest.approx((1.32 + 1.07) / 2)
    by_mass = Material.from_mass_fractions({ec: 0.5, dmc: 0.5})
    assert by_mass.density == pytest.approx(1 / (0.5 / 1.32 + 0.5 / 1.07))
    sol = Material.solution(blend, "LiPF6", 1.0, density=1.25)
    w_salt = 151.905 / 1250.0
    assert sol.mass_fractions["P"] == pytest.approx(w_salt * 30.974 / 151.905, rel=1e-3)
    assert sum(sol.mass_fractions.values()) == pytest.approx(1.0)


def test_material_validation():
    with pytest.raises(ValueError):
        Material("C", -1)
    with pytest.raises(ValueError):
        Material("C", 1.0, mass_fractions={"C": 1})


# ------------------------------------------------------------------------------ beam

def test_spectrum_constructors():
    e = np.linspace(5e3, 30e3, 501)
    s = Spectrum.from_density(e, np.ones_like(e) * 1e6)
    assert s.total_flux == pytest.approx(1e6 * 25e3)
    s01 = Spectrum.from_per_bandwidth(e, np.ones_like(e))
    assert s01.total_flux == pytest.approx(1e3 * math.log(30 / 5), rel=1e-3)
    assert s.scaled_to(1.0).total_flux == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Spectrum(np.array([1e4, 1e4]), np.array([1.0, 1.0]))


def test_beam_profiles():
    b = Beam.monochromatic(20e3, 1e10, 2, 3, profile="gaussian")
    assert b.peak_factor == pytest.approx(4 * math.log(2) / math.pi)
    assert b.mean_factor == pytest.approx(b.peak_factor / 2)
    legacy = Beam(beamEnergy=27000, photonFlux=1e10, beamWidth=10, beamHeight=20, exposure=5)
    assert legacy.beamEnergy == 27000 and legacy.exposure == 5 and legacy.profile == "tophat"


# ----------------------------------------------------------------------------- dose

@pytest.fixture
def cell():
    nmc = Material("LiNi0.8Mn0.1Co0.1O2", 4.8, "NMC")
    ely = Material("C3H6O3", 1.15, "electrolyte")      # one instance, reused
    return Stack([
        Layer.solid("Cu", Material("Cu", 8.96, "copper"), 10),
        Layer.porous("anode", Material("C", 2.3, "graphite"), 0.40, ely, 54),
        Layer.porous("cathode", nmc, 0.28, ely, 32),
    ])


def test_thin_sample_dose_is_fluence_times_mu_en(cell):
    beam = Beam.monochromatic(27e3, 1e10, 100, 100)
    st = Stack([Layer.solid("foil", Material("C", 2.0), 0.01)])
    d = st.compute(beam).components[0].dose_rate
    fluence = 1e10 / (100e-4 * 100e-4)
    expected = fluence * 27e3 * 1.602176634e-19 * xd.mu_en_rho("C", 27e3)[0] * 1e3
    assert d == pytest.approx(expected, rel=1e-6)


def test_energy_conservation_in_attenuation_mode(cell):
    beam = Beam(Spectrum.from_density(np.linspace(8e3, 30e3, 300), np.full(300, 1e7)), 50, 50)
    r = cell.compute(beam, deposition="attenuation")
    p_abs = sum(c.absorbed_power_w for c in r.components)
    p_out = r.transmitted.power_w
    assert p_abs + p_out == pytest.approx(beam.spectrum.power_w, rel=1e-10)
    r_en = cell.compute(beam)
    assert sum(c.absorbed_power_w for c in r_en.components) < p_abs


def test_mu_en_lower_than_attenuation_and_ratio(cell):
    beam = Beam.monochromatic(27e3, 1e10, 100, 100)
    a = cell.compute(beam, deposition="attenuation")
    b = cell.compute(beam)
    for ca, cb in zip(a.components, b.components):
        ratio = cb.dose_rate / ca.dose_rate
        expected = (cb.material_obj.mu_en_rho(27e3) / cb.material_obj.mu_rho(27e3))[0]
        assert ratio == pytest.approx(expected, rel=1e-9)
    assert b.get("anode", "graphite").dose_rate / a.get("anode", "graphite").dose_rate < 0.4


def test_material_reuse_gives_per_layer_results(cell):
    r = cell.compute(Beam.monochromatic(27e3, 1e10, 100, 100))
    d1 = r.get("anode", "electrolyte").dose_rate
    d2 = r.get("cathode", "electrolyte").dose_rate
    assert d1 != d2 and d1 > 0 and d2 > 0


def test_mixed_dose_is_mass_weighted(cell):
    r = cell.compute(Beam.monochromatic(27e3, 1e10, 100, 100))
    L = r.get("cathode")
    num = sum(c.dose_rate * c.volume_fraction * c.density for c in L.components)
    den = sum(c.volume_fraction * c.density for c in L.components)
    assert L.mixed_dose_rate == pytest.approx(num / den, rel=1e-9)


def test_white_beam_monochromatic_limit():
    st = Stack([Layer.porous("c", Material("LiNi0.8Mn0.1Co0.1O2", 4.8), 0.3, Material("C3H6O3", 1.15), 30)])
    eg = np.linspace(19950, 20050, 201)
    narrow = Spectrum.from_density(eg, np.exp(-0.5 * ((eg - 2e4) / 5) ** 2)).scaled_to(1e10)
    a = st.compute(Beam(narrow, 100, 100)).components[0].dose_rate
    b = st.compute(Beam.monochromatic(2e4, 1e10, 100, 100)).components[0].dose_rate
    assert a == pytest.approx(b, rel=1e-5)


def test_beam_hardening():
    e = np.linspace(8e3, 30e3, 200)
    beam = Beam(Spectrum.from_density(e, np.ones_like(e)), 10, 10)
    st = Stack([Layer.solid("Cu", Material("Cu", 8.96), 20), Layer.solid("Al", Material("Al", 2.7), 10)])
    r = st.compute(beam)
    assert r.spectra_in[1].mean_energy_ev > r.spectra_in[0].mean_energy_ev


def test_incidence_angle_scales_path():
    beam = Beam.monochromatic(15e3, 1e10, 100, 100)
    m = Material("Al", 2.7)
    t60 = Stack([Layer.solid("Al", m, 20)], incidence_angle_deg=60).compute(beam).layers[0].transmission
    t0 = Stack([Layer.solid("Al", m, 40)]).compute(beam).layers[0].transmission
    assert t60 == pytest.approx(t0, rel=1e-9)
    assert Stack([Layer.solid("Al", m, 1)], grazing_angle_deg=40).incidence_angle_deg == pytest.approx(50)


def test_depth_profile_and_semi_infinite():
    beam = Beam.monochromatic(10e3, 1e10, 10, 10)
    st = Stack([Layer.solid("bulk", Material("Si", 2.33), np.inf)], grazing_angle_deg=40)
    r = st.compute(beam)
    c = r.components[0]
    assert c.dose_rate == pytest.approx(c.dose_rate_max * (1 - math.exp(-1)), rel=1e-9)
    z, d = r.depth_profile("bulk", "Si")
    assert d[0] == pytest.approx(c.dose_rate_max, rel=1e-9)
    assert np.all(np.diff(d) < 0)
    assert r.transmission == 0


def test_layer_validation():
    m = Material("C", 2.0)
    with pytest.raises(ValueError):
        Layer("x", components=[(m, 0.7), (m, 0.5)], thickness_um=1)
    with pytest.raises(ValueError):
        Layer("x", solidMaterial=m, liquidMaterial=None, thickness=0.01, porosity=0.3)
    with pytest.raises(ValueError):
        Layer("x", solidMaterial=m, thickness=0.01, porosity=1.5)
    with pytest.raises(ValueError):
        Stack([Layer.solid("a", m, np.inf), Layer.solid("b", m, 1)])


# --------------------------------------------------------------------------- escape

def test_electron_range_water():
    # ESTAR CSDA range of water at 20 keV: 8.57e-4 g/cm2
    r = escape.electron_range_g_cm2(Material("H2O", 1.0).mass_fractions, 20.0)[0]
    assert r == pytest.approx(8.57e-4, rel=0.06)


def test_retention_limits():
    big_m, big_p = escape.retention([1.0], 1e4, 1e4, "tophat")
    small_m, small_p = escape.retention([10.0], 0.01, 0.01, "gaussian")
    assert big_m[0] > 0.999 and big_p[0] > 0.999
    assert small_m[0] < 0.01 and small_p[0] < 0.01
    m, _ = escape.retention(np.geomspace(0.1, 10, 20), 1, 1, "gaussian")
    assert np.all(np.diff(m) < 0)


def test_escape_changes_only_small_beams():
    st = Stack([Layer.solid("Si", Material("Si", 2.33), 50)])
    big = Beam.monochromatic(20e3, 1e10, 500, 500)
    small = Beam.monochromatic(20e3, 1e10, 0.5, 0.5, profile="gaussian")
    rb = st.compute(big, electron_escape=True).components[0]
    rs = st.compute(small, electron_escape=True).components[0]
    assert rb.retention_mean > 0.99
    assert rs.retention_mean < 0.8
    assert rs.dose_rate == pytest.approx(rs.kerma_rate * rs.retention_mean, rel=1e-9)


# ------------------------------------------------------------------------- exposure

def test_exposure_schedule():
    s = ExposureSchedule.periodic(on_s=10, period_s=30, n=3)
    assert s.total_exposure_s == 30 and s.span_s == 70
    t, d = s.dose_history(2.0)
    assert d[-1] == pytest.approx(60.0)
    assert s.mean_dose_rate(2.0) == pytest.approx(2.0 * 30 / 70)
    with pytest.raises(ValueError):
        ExposureSchedule(((0, 10), (5, 10)))


# --------------------------------------------------------------------------- legacy

def test_legacy_notebook_runs_and_matches_v0_within_rayleigh():
    xr = Beam(beamEnergy=27000, photonFlux=2.44E10, beamWidth=630, beamHeight=110, exposure=1050 * 27 * .375)
    layers = [
        Layer(name="Cu collector", solidMaterial=Material(name="copper", formula="Cu", density=8.96),
              liquidMaterial=None, thickness=0.010, porosity=0.0),
        Layer(name="cathode", solidMaterial=Material(name="NMC", formula="LiNi0.8Mn0.1Co0.1O2", density=4.8),
              liquidMaterial=Material(name="electrolyte", formula="C3H6O3", density=1.15),
              thickness=0.032, porosity=0.28),
    ]
    cell = Stacking(layers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cell.get_dose(xr, deposition="attenuation")
    # v0 (Henke + Klein-Nishina, no Rayleigh) gave 21762 kGy for Cu; NIST mu is 3 % higher
    assert layers[0].solidMaterial.dose_kGy == pytest.approx(21762, rel=0.05)
    assert 0 < layers[1].solidMaterial.absorption < 0.1
    cell.get_dose(xr)            # mu_en
    assert layers[0].solidMaterial.dose_kGy == pytest.approx(21762 * 0.87, rel=0.03)


def test_spectral_contribution_sums_and_time_to_dose(cell):
    e = np.linspace(8e3, 30e3, 100)
    beam = Beam(Spectrum.from_density(e, np.full(100, 1e7)), 20, 20, profile="gaussian")
    r = cell.compute(beam, electron_escape=True)
    c = r.get("cathode", "NMC")
    en, contrib = r.spectral_contribution("cathode", "NMC")
    assert contrib.sum() == pytest.approx(c.dose_rate, rel=1e-9)
    _, cpk = r.spectral_contribution("cathode", "NMC", peak=True)
    assert cpk.sum() == pytest.approx(c.dose_rate_max, rel=1e-9)
    _, mixed = r.spectral_contribution("cathode")
    assert mixed.sum() == pytest.approx(r.get("cathode").mixed_dose_rate, rel=1e-9)
    assert r.time_to_dose(1e6, "cathode", "NMC") == pytest.approx(1e6 / c.dose_rate_max)


def test_raster_factor():
    from dose_estimation import raster_factor
    assert raster_factor(Beam.monochromatic(2e4, 1, 10, 10), 10, 10) == pytest.approx(1.0)
    g = Beam.monochromatic(2e4, 1, 1, 1, profile="gaussian")
    assert raster_factor(g, 1, 1) == pytest.approx(1 / g.mean_factor)
