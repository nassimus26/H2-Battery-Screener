"""Sealed two-gas batteries: generation, thermodynamics, cells, sizing."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from h2screener import gases, options, storage
from h2screener.constants import F

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def measured():
    """Cells from measured molecules only (no xTB energies)."""
    return gases.screen(cache={})


def test_generation_is_exhaustive_and_physical():
    mols = set(gases.enumerate_molecules(3))
    assert {"C", "N", "O", "CC", "C=O", "O=C=O", "C#N", "ClCl", "O=S=O"} <= mols
    assert not any("#S" in s or "S#" in s for s in mols)            # no sulfur triple bonds
    assert all(not gases._radical(s) for s in mols)


def test_joback_boiling_point_for_a_mid_size_molecule():
    assert gases.joback_tb("CC(C)=O") == pytest.approx(329.2, abs=15)    # acetone


def test_couple_potentials_follow_measured_gibbs_energies(measured):
    _, sp = measured
    cps = {(c.ox, c.red): c for c in gases.couples(sp)}
    assert cps[("O=O", "O")].e0 == pytest.approx(1.229, abs=0.002)
    co2_ch4 = cps[("O=C=O", "C")]                                    # CO2 + 4 H2 -> CH4 + 2 H2O
    assert co2_ch4.n_e == 8 and co2_ch4.water == 2
    assert co2_ch4.e0 == pytest.approx((-50.5 + 2 * -237.1 + 394.4) * -1000 / (8 * F), abs=1e-6)


def test_hydrogen_oxygen_cell_energy_per_mole_of_gas(measured):
    cells, _ = measured
    ref = gases.reference_h2_o2(cells)
    assert ref.voltage == pytest.approx(1.229, abs=0.002)
    assert ref.wh_per_mol_gas == pytest.approx(4 * F * 1.229 / 3600 / 3, rel=1e-3)
    assert ref.recharge == "demonstrated"
    methane = next(c for c in cells if c.name == "2 O2(g) + CH4(g) -> CO2(g) + 2 H2O(l)")
    assert methane.voltage == pytest.approx(1.059, abs=0.002)
    assert methane.wh_per_mol_gas > ref.wh_per_mol_gas               # 8 electrons for 4 moles of gas
    assert methane.recharge == "one direction"


def test_condensed_gas_needs_a_lighter_tank():
    t = tomllib.load(open(ROOT / "targets.toml", "rb"))
    permanent = storage.gas_tank("Xe", 100.0, 298.15, 2e5, 350e5, t, tb_k=0.0)[0]
    condensed = storage.gas_tank("Xe", 100.0, 298.15, 2e5, 350e5, t, tb_k=260.0)[0]
    assert condensed < 0.2 * permanent


def test_sealed_gas_battery_is_sized_with_every_gas_tank(measured):
    cells, sp = measured
    t = tomllib.load(open(ROOT / "targets.toml", "rb"))
    ref = gases.reference_h2_o2(cells)
    cfgs = options.gas_configurations(ref, sp, t, np.random.default_rng(0))
    assert len(cfgs) == 2
    for c in cfgs:
        assert c.cost_parts.get("H2 storage", 0) > 0 and c.cost_parts.get("O2 storage", 0) > 0
        assert c.wh_kg > 140 and c.store.burst_tnt_kg > 0
