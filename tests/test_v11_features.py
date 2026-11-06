"""Tests for the v1.1 additions: atomic data, fluorescence reabsorption, electron Monte Carlo,
Z-dependent escape, thermal estimates and uncertainty propagation."""

import math

import numpy as np
import pytest

from dose_estimation import Beam, Layer, Material, Spectrum, Stack
from dose_estimation import atomic, electron_mc as emc, escape, thermal, xray_data as xd
from dose_estimation.uncertainty import Uncertain, propagate, propagate_stack


# ---------------------------------------------------------------------------- atomic data

def test_atomic_tables():
    edge, omega, jump = atomic.levels("Ni")["K"]
    assert edge == pytest.approx(8333.0) and omega == pytest.approx(0.412, abs=0.01) and jump > 7
    ka = atomic.line_groups("Cu", "K")
    assert ka[0][0] == pytest.approx(8040, abs=10) and sum(i for _, i in ka) == pytest.approx(1.0)
    assert atomic.binding_energy_below("C", 20e3) == (pytest.approx(284.2), "K")
    assert max(e for e, _ in atomic.fluorescence_lines("Cu", 8000.0)) < 1100    # below the K edge: L lines only


@pytest.mark.parametrize("el, e", [("Cu", 20e3), ("Ni", 15e3), ("Ge", 20e3), ("Zr", 25e3)])
def test_photo_coefficient_from_mu_en_is_consistent(el, e):
    x = atomic.fluorescence_energy_per_absorption(el, e)
    tau = xd.mu_en_rho(el, e)[0] / (1 - x / e)
    assert 0.9 < tau / xd.mu_rho(el, e)[0] < 1.0


# ---------------------------------------------------------------------------- fluorescence

def test_fluorescence_matches_analytic_escape_formula():
    cu = Material("Cu", 8.96)
    e = 20e3
    r = Stack([Layer.solid("Cu", cu, np.inf)]).compute(Beam.monochromatic(e, 1e10, 1e5, 1e5),
                                                       fluorescence=True, mc_photons=100000)
    mu_p = cu.mu(e)[0]
    num = 0.0
    for ef, y in atomic.fluorescence_lines("Cu", e):
        if ef < 1000:
            num += y * ef
            continue
        mu_f = cu.mu(ef)[0]
        esc = 0.5 * (1 - (mu_f / mu_p) * math.log(1 + mu_p / mu_f))       # classic escape fraction
        num += y * ef * (1 - esc) * cu.mu_en_rho(ef)[0] / cu.mu_rho(ef)[0]
    x = atomic.fluorescence_energy_per_absorption("Cu", e)
    tau = xd.mu_en_rho("Cu", e)[0] / (1 - x / e)
    expected = tau * num / (e * xd.mu_en_rho("Cu", e)[0])
    assert r.components[0].fluorescence_ratio == pytest.approx(expected, rel=0.01)


def test_fluorescence_bookkeeping_and_small_beam():
    nmc = Material("LiNi0.8Mn0.1Co0.1O2", 4.8, "NMC")
    st = Stack([Layer.solid("Cu", Material("Cu", 8.96), 10),
                Layer.porous("cathode", nmc, 0.3, Material("C3H6O3", 1.15, "ely"), 30)])
    big = st.compute(Beam.monochromatic(27e3, 1e10, 600, 100), fluorescence=True)
    rep = big.fluorescence
    assert rep.emitted_w == pytest.approx(rep.reabsorbed_in_stack_w + rep.escaped_w + rep.local_w + rep.skipped_w,
                                          rel=1e-9)
    ely = big.get("cathode", "ely")
    assert ely.fluorescence_ratio > 0.2                     # Ni K fluorescence feeds the electrolyte
    assert ely.dose_rate == pytest.approx(ely.kerma_rate * (1 + ely.fluorescence_ratio), rel=1e-9)
    small = st.compute(Beam.monochromatic(27e3, 1e10, 0.5, 0.5, profile="gaussian"), fluorescence=True)
    assert small.get("cathode", "ely").fluorescence_ratio < 0.02
    e, c = big.spectral_contribution("cathode", "ely")
    assert c.sum() == pytest.approx(ely.dose_rate, rel=1e-9)
    with pytest.raises(ValueError):
        st.compute(Beam.monochromatic(27e3, 1e10, 10, 10), deposition="attenuation", fluorescence=True)


# ---------------------------------------------------------------------------- electron Monte Carlo

@pytest.mark.parametrize("sym, rho, tol", [("C", 2.0, 0.15), ("Al", 2.7, 0.10), ("Cu", 8.96, 0.10)])
def test_mc_backscatter_against_measurements(sym, rho, tol):
    med = emc.ElectronMedium({sym: 1.0}, rho)
    eta = emc.backscatter_coefficient(med, 20.0, n=2000)
    assert eta == pytest.approx(emc.hunger_kuchler(med.z[0], 20.0), rel=tol)


def test_mc_csda_range_water():
    w = Material("H2O", 1.0)
    assert emc.ElectronMedium(w.mass_fractions, 1.0).csda_range_g_cm2(20.0) == pytest.approx(8.57e-4, rel=0.15)


def test_mc_retention_limits_and_agreement():
    m = Material("LiNi0.8Mn0.1Co0.1O2", 4.8)
    big = emc.retention_mc(m.mass_fractions, 4.8, 20e3, 100, 100, "gaussian", None, n_electrons=300)
    assert big.mean > 0.99
    mc = emc.retention_mc(m.mass_fractions, 4.8, 20e3, 0.5, 0.5, "gaussian", None, "isotropic", n_electrons=1500)
    an = escape.component_retention(m.mass_fractions, m.mass_fractions, 4.8, [20e3], 0.5, 0.5, "gaussian")[0][0]
    assert an == pytest.approx(mc.mean, rel=0.1)


def test_detour_factor_depends_on_z():
    assert escape.detour_factor_for(Material("C3H6O3", 1.0).mass_fractions) == pytest.approx(0.87, abs=0.05)
    assert escape.detour_factor_for({"Au": 1.0}) == pytest.approx(0.36, abs=0.03)


def test_compute_with_mc_escape():
    st = Stack([Layer.solid("Ge", Material("Ge", 5.32, "Ge"), 20)])
    beam = Beam.monochromatic(20e3, 1e10, 0.5, 0.5, profile="gaussian")
    an = st.compute(beam, electron_escape=True).components[0]
    mc = st.compute(beam, electron_escape="mc", mc_electrons=800).components[0]
    assert mc.retention_mean == pytest.approx(an.retention_mean, rel=0.12)
    with pytest.raises(ValueError):
        st.compute(beam, electron_escape="foo")


# ---------------------------------------------------------------------------- thermal

def test_halfspace_rise_analytic():
    rng = np.random.default_rng(0)
    p, k, s = 1e-3, 60.0, 0.5
    pts = np.column_stack([rng.normal(0, s, 100000), rng.normal(0, s, 100000), np.zeros(100000)])
    lax = p / (2 * math.sqrt(2 * math.pi) * k * s * 1e-6)
    assert thermal.halfspace_rise(pts, p, k, [[0, 0, 0]])[0] == pytest.approx(lax, rel=0.02)


def test_plate_rise_analytic():
    p, kk, a, r = 1e-3, 4e-3, 1.0, 1e4
    side = math.sqrt(math.pi) * a
    disc = thermal.plate_center_rise(p, kk, "tophat", side, side, sink_radius_um=r)
    assert disc == pytest.approx(p / (4 * math.pi * kk) * (1 + 2 * math.log(r / a)), rel=1e-6)


def test_temperature_rise_on_results():
    nmc = Material("LiNi0.8Mn0.1Co0.1O2", 4.8, "NMC")
    cell = Stack([Layer.solid("Cu", Material("Cu", 8.96, "Cu"), 10),
                  Layer.porous("cathode", nmc, 0.3, Material("C3H6O3", 1.15, "ely"), 30)])
    res = cell.compute(Beam.monochromatic(27e3, 2.44e10, 630, 300))     # footprint >> 40 um stack
    est = thermal.temperature_rise(res, {"Cu": 401, "cathode": 1.0}, sink_radius_um=1e4)
    p_cath = sum(c.absorbed_power_w for c in res.get("cathode").components)
    q = p_cath / (630e-6 * 300e-6 * 30e-6)
    assert est.through_thickness_k["cathode"] == pytest.approx(q * (30e-6) ** 2 / 8, rel=1e-9)
    assert 0 < est.lateral_k < 1 and est.total_k > est.lateral_k
    micro = cell.compute(Beam.monochromatic(27e3, 1e11, 0.5, 0.5, profile="gaussian"))
    em = thermal.temperature_rise(micro, {"Cu": 401, "cathode": 1.0}, sink_radius_um=1e4)
    assert not em.through_thickness_k and em.near_field_k > 0 and em.total_k > em.near_field_k
    bulk = Stack([Layer.solid("Ge", Material("Ge", 5.32, "Ge"), np.inf)], grazing_angle_deg=40)
    rb = bulk.compute(Beam.monochromatic(20e3, 1e12, 0.5, 0.5, profile="gaussian"))
    eb = thermal.temperature_rise(rb, 60.0)
    p = rb.components[0].absorbed_power_w
    # bounded by a surface Gaussian source of the same power (all heat at the surface)
    s = rb.beam.sigma_um[0]
    assert 0 < eb.bulk_k < p / (2 * math.sqrt(2 * math.pi) * 60.0 * s * 1e-6) * 1.05


# ---------------------------------------------------------------------------- uncertainty

def test_propagate_flux_only_and_sensitivity():
    st = Stack([Layer.solid("Si", Material("Si", 2.33, "Si"), 10)])

    def model(flux):
        return st.compute(Beam.monochromatic(15e3, flux, 10, 10))

    u = propagate(model, {"flux": Uncertain(1e10, rel=0.2)}, n=2000, seed=1)
    row = u.table().iloc[0]
    assert row["rel_sigma"] == pytest.approx(0.2, rel=0.1)
    assert u.sensitivity().loc["Si/Si", "flux"] > 0.99


def test_propagate_stack_porosity_does_not_change_component_dose():
    nmc = Material("LiNi0.8Mn0.1Co0.1O2", 4.8, "NMC")
    st = Stack([Layer.porous("cathode", nmc, 0.3, Material("C3H6O3", 1.15, "ely"), 5)])
    u = propagate_stack(st, Beam.monochromatic(27e3, 1e10, 100, 100), flux_rel=0.0, size_rel=0.0,
                        porosity_sd=0.05, n=200, seed=2)
    t = u.table().set_index("quantity")
    assert t.loc["cathode/NMC", "rel_sigma"] < 1e-3
    assert t.loc["cathode/(mixed)", "rel_sigma"] > 0.01
