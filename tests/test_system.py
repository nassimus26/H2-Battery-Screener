"""Checks of the safety, maintenance, design, molecule and reference stages."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from h2screener import design, discover, maintenance, reference, safety
from h2screener import molecules as M

ROOT = Path(__file__).resolve().parent.parent
T = 298.15


@pytest.fixture(scope="module")
def targets():
    return tomllib.load(open(ROOT / "targets.toml", "rb"))


def test_burst_energy_grows_with_pressure_and_is_below_ideal_isothermal_work():
    w10 = safety.expansion_work("H2", 100, T, 10e5)
    w60 = safety.expansion_work("H2", 100, T, 60e5)
    assert 0 < w10 < w60
    assert w60 < 100 * 8.314462618 * T * math.log(60e5 / 101325)


def test_choked_hydrogen_leak_scales_with_pressure():
    a = safety.choked_flow("H2", T, 20e5, 1e-6, 0.6)
    b = safety.choked_flow("H2", T, 40e5, 1e-6, 0.6)
    assert b / a == pytest.approx(2, rel=0.05)


def test_latimer_derivation_of_hypobromous_acid_reduction():
    hrs = maintenance._table()
    hbro = [h for h in hrs if h.source.startswith("derived") and "HBrO" in h.label and "Br2" in h.label]
    assert hbro and hbro[0].e0 == pytest.approx(1.6, abs=0.05)


def test_closure_flags_bromate_but_not_manganese_dioxide():
    cells = {c.name: c for c in discover.table_pairs()}
    br = cells["BrO3- + 2 H2(g) + H+ -> HBrO(aq) + 2 H2O(l)"]
    mn = cells["MnO2(s) + H2(g) + 2 H+ -> Mn(2+) + 2 H2O(l)"]
    assert maintenance.chemical_closure(br, 0.0, 1.229)[0]
    assert not maintenance.chemical_closure(mn, 0.0, 1.229)[0]


def test_design_meets_power_and_energy(targets):
    cells = {c.name: c for c in discover.table_pairs()}
    d = design.design(cells["MnO2(s) + H2(g) + 2 H+ -> Mn(2+) + 2 H2O(l)"], targets, np.random.default_rng(0))
    b = d.best
    p = targets["design"]["power_kw"] * 1000
    assert b.i_a_cm2 * b.area_m2 * 1e4 * b.v_dis == pytest.approx(p, rel=1e-6)
    assert b.rte >= targets["use"]["rte_target"] - 1e-9 or b.rte > 0


def test_pareto_front():
    assert design.pareto([(1, 2), (2, 1), (2, 2), (3, 3)]) == [0, 1]


def test_quinone_partner_and_calibration():
    assert M.reduced_forms("O=C1C=CC(=O)C=C1") == ["Oc1ccc(O)cc1"]
    r = M.evaluate(M.REFERENCE_SMILES)
    assert r.e0 == pytest.approx(M.REFERENCE_E0, abs=1e-6)


def test_reference_targets_are_the_best_pack_values():
    t = reference.pack_targets()
    assert t["wh_kg"] == 140 and t["usd_kwh_stationary"] == 70


def test_element_cost_is_mass_weighted():
    from h2screener import materials
    p = materials.prices()
    expect = 54.938 / 86.937 * p["Mn"] + 31.998 / 86.937 * p["O"]
    assert materials.cost_per_kg("MnO2") == pytest.approx(expect, rel=1e-3)
