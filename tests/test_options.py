"""Storage options and recharge modes."""

import csv
import tomllib
from pathlib import Path

import numpy as np
import pytest

from h2screener import discover, options, realgas, storage
from h2screener.constants import M_H2

ROOT = Path(__file__).resolve().parent.parent
T = 298.15


@pytest.fixture(scope="module")
def targets():
    return tomllib.load(open(ROOT / "targets.toml", "rb"))


def test_van_t_hoff_matches_the_database_25c_pressures():
    checked = close = 0
    for row in csv.DictReader(open(storage.HYDPARK, encoding="utf-8")):
        try:
            a = storage.Alloy(row["Composition_Formula"], 1.0, float(row["Heat_of_Formation_kJperMolH2"]),
                              float(row["Entropy_of_Formation_JperMolH2perK"]), "", "")
            ref = float(row["Equilibrium_Pressure_25C"])
        except ValueError:
            continue
        if ref > 1e-6:
            r = a.p_eq(T) / 101325 / ref
            assert abs(r - 1) < 0.10, row["Composition_Formula"]
            checked += 1
            close += abs(r - 1) < 0.02
    # a few supplemented entries use a slightly different convention (3-7 % off)
    assert checked > 100 and close / checked > 0.95


def test_compression_work_grows_with_pressure():
    w350 = storage.compression_work_j_per_mol(T, 30e5, 350e5, 3, 0.7)
    w700 = storage.compression_work_j_per_mol(T, 30e5, 700e5, 3, 0.7)
    assert 0 < w350 < w700


def test_composite_tank_reproduces_published_weight_fraction(targets):
    sigma = options.composite_stress_mpa(targets) * 1e6
    rf, sm = targets["refuel"], targets["system"]
    p = 700e5
    rho = realgas.molar_density("H2", T, p) * M_H2
    wall_per_kg = sm["shape_factor"] * rf["composite_wall_density_kg_m3"] * p / (sigma * rho)
    assert 1 / (1 + wall_per_kg) == pytest.approx(0.057, rel=1e-6)


def test_every_option_is_sized_for_manganese_dioxide(targets):
    cells = {c.name: c for c in discover.table_pairs()}
    cfgs = options.configurations(cells["MnO2(s) + H2(g) + 2 H+ -> Mn(2+) + 2 H2O(l)"], targets,
                                  np.random.default_rng(0))
    kinds = {c.store.kind for c in cfgs}
    assert {"tank", "piston", "hydride"} <= kinds
    hyd = [c for c in cfgs if c.store.kind == "hydride"]
    tank = [c for c in cfgs if c.store.kind == "tank"][0]
    assert all(h.store.burst_tnt_kg < tank.store.burst_tnt_kg for h in hyd)   # low constant pressure


def test_refuel_mode_counts_the_whole_chain(targets):
    cfgs = options.refuel_configurations(discover.table_pairs(), targets, np.random.default_rng(0))
    assert cfgs and all(0 < c.rte < 0.83 for c in cfgs)         # below a lone H2-O2 cell's ideal
    assert all(c.recharge_h < 1 for c in cfgs)                  # filling, not charging
