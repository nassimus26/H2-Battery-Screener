"""Checks of the physical laws and bookkeeping the screener relies on."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from h2screener import discover, realgas, system
from h2screener.constants import F, R
from h2screener.performance import CellModel, stokes_einstein
from h2screener.estimate import cell_params
from h2screener.reactions import parse_side, parse_species, reaction_balanced

ROOT = Path(__file__).resolve().parent.parent
T = 298.15


@pytest.fixture(scope="module")
def targets():
    return tomllib.load(open(ROOT / "targets.toml", "rb"))


@pytest.fixture(scope="module")
def cells():
    return {c.name: c for c in discover.table_pairs()}


def test_species_parsing():
    assert parse_species("Fe(3+)").charge == 3
    assert parse_species("[Fe(CN)6](4-)").charge == -4
    assert parse_species("MnO4-").charge == -1
    s = parse_species("Br2(l)")
    assert (s.formula, s.phase, s.charge) == ("Br2", "l", 0)
    assert parse_species("HO•") is None


def test_every_generated_reaction_is_balanced(cells):
    assert len(cells) > 100
    for c in cells.values():
        assert reaction_balanced(c.reactants, c.products), c.name
        assert c.e0 > 0


def test_hydrogen_is_nearly_ideal_at_one_bar_and_not_at_350():
    assert realgas.fugacity("H2", T, 1e5) / 1e5 == pytest.approx(1.0, abs=2e-3)
    ideal = 350e5 / (R * T)
    assert realgas.molar_density("H2", T, 350e5) < 0.85 * ideal   # Z > 1.15


def test_chlorine_is_liquid_when_stored_at_10_bar():
    assert realgas.condenses("Cl2", T, 10e5)
    assert not realgas.condenses("H2", T, 700e5)


def test_nernst_pressure_shift(cells, targets):
    """Raising H2 from 1 to 60 bar adds (RT/2F) ln(f60/f1) per H2 per 2 e-."""
    c = cells["Br2(aq) + H2(g) -> 2 Br- + 2 H+"]
    lo = CellModel(c, T, 0.999e5, 1e5, cell_params(c, T, 0.999e5, 1e5, targets))
    hi = CellModel(c, T, 59.99e5, 60e5, cell_params(c, T, 59.99e5, 60e5, targets))
    s = np.array([0.5])
    shift = hi.ocv(s)[0] - lo.ocv(s)[0]
    expect = R * T / (2 * F) * math.log(realgas.fugacity("H2", T, 60e5) / realgas.fugacity("H2", T, 1e5))
    assert shift == pytest.approx(expect, rel=1e-3)
    assert 0.05 < shift < 0.06


def test_stokes_einstein_gives_aqueous_ion_diffusivity():
    d = stokes_einstein(T, 0.25e-9)
    assert 0.5e-9 < d < 2e-9      # m2/s, the range of common aqueous ions


def test_vessel_wall_mass_follows_membrane_stress(cells, targets):
    c = cells["Br2(aq) + H2(g) -> 2 Br- + 2 H+"]
    sp = system.system_point(c, 1.0, 2e5, 60e5, T, targets)
    s = targets["system"]
    n_h2 = 1000 * 3600 / 1.0 / (2 * F)
    v = realgas.swing_volume("H2", n_h2, T, 2e5, 60e5)
    expect = s["shape_factor"] * s["wall_density_kg_m3"] * 60e5 * v / (s["wall_allowable_stress_mpa"] * 1e6)
    assert sp.parts["vessel"] == pytest.approx(expect)


def test_reactant_specific_energy_of_mno2_cell(cells):
    c = cells["MnO2(s) + H2(g) + 2 H+ -> Mn(2+) + 2 H2O(l)"]
    wh_kg = c.e0 * c.n * F / c.active_mass() / 3600
    # MnO2 (86.94 g) + H2 (2.02 g), 2 e-, 1.224 V
    assert wh_kg == pytest.approx(1.224 * 2 * F / 88.95e-3 / 3600, rel=2e-3)
