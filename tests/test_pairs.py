"""Two-molecule cells."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from h2screener import options, pairs
from h2screener import molecules as M
from h2screener.constants import F
from h2screener.reactions import reaction_balanced

ROOT = Path(__file__).resolve().parent.parent


def mol(smiles, e0, sol=1.0):
    r = M.MolResult(M.canonical(smiles), M.reduced_forms(smiles)[0], 0.0, e0)
    r.mw, r.solubility_mol_l, r.sa = M.descriptors(smiles)["mw"], sol, 2.0
    return r


def test_pair_voltage_and_energy_per_litre():
    v, whl, _ = pairs.pair_value(0.1, 1.0, 200, 0.9, 1.0, 100)
    assert v == pytest.approx(0.8)
    assert whl == pytest.approx(0.8 * 2 * F / 3600 / 2)        # two 1 M tanks per mole of electrons-pair


def test_water_window_and_headroom_reject():
    assert pairs.pair_value(-0.5, 1, 100, 0.8, 1, 100)[1] == 0      # negative would make H2
    assert pairs.pair_value(0.1, 1, 100, 1.6, 1, 100)[1] == 0       # positive would make O2


def test_pair_cell_is_balanced_and_gas_free():
    n = mol("O=C1c2ccccc2C(=O)c2ccccc21", 0.1)
    p = mol("O=C1C=CC(=O)C=C1", 0.7)
    c = pairs.as_cell(pairs.make_pair(n, p))
    assert reaction_balanced(c.reactants, c.products)
    assert not c.gas_terms() and c.h2_consumed() == 0


def test_pair_sizing_has_no_burst_energy():
    t = tomllib.load(open(ROOT / "targets.toml", "rb"))
    n = mol("O=C1c2ccccc2C(=O)c2ccccc21", 0.1, 2.0)
    p = mol("O=C1C=CC(=O)C=C1", 0.75, 2.0)
    c = options.pair_configuration(pairs.make_pair(n, p), t, np.random.default_rng(0))
    assert c is not None and c.store.burst_tnt_kg == 0 and c.wh_kg > 0


def test_price_ceiling_follows_mass_per_kwh():
    kg = pairs.kg_per_kwh(200, 1.0)
    assert kg == pytest.approx(1000 * 3600 / (2 * F) * 0.2 * 1.2)


def test_one_molecule_with_two_steps_makes_a_symmetric_cell():
    r = M.MolResult("X", "XH2", 0.0, 1.25, steps=[1.25, 0.22], forms=["XH2", "XH4"])
    r.reduced, r.mw, r.solubility_mol_l, r.sa = "XH2", 200.0, 1.0, 2.0
    pr = pairs.make_pair(r, r, kn=1, kp=1)
    assert pr.voltage == pytest.approx(1.03)
    assert pr.score > 0 and "same molecule" in pr.notes[0]


def test_two_electron_steps_double_capacity():
    one = pairs.pair_value([0.1], 1, 200, [0.8], 1, 200)[1]
    two = pairs.pair_value([0.1], 1, 200, [0.8, 0.8], 1, 200)[1]
    assert two > one


def test_groups_that_fall_apart_in_water_are_flagged():
    from h2screener import molecules as M
    assert any("aminal" in x for x in M.instability("CNCN=c1c2nc(=O)[nH]c(=O)c-2nc1=O"))
    assert any("imine" in x for x in M.instability("CNCN=c1c2nc(=O)[nH]c(=O)c-2nc1=O"))
    assert any("hydrazine" in x for x in M.instability("CNN1CN=C2C(=O)NC(=O)N=C21"))
    assert any("nitroso" in x for x in M.instability("NC(=O)C(=O)N=O"))
    for good in ("O=C1C=CC(=O)C=C1", "O=C1c2ccccc2C(=O)c2ccccc12", "O=c1[nH]c(=O)c2nc3ccccc3nc2[nH]1",
                 "C[n+]1ccc(-c2cc[n+](C)cc2)cc1", "CC1(C)CC(O)CC(C)(C)N1[O]"):
        assert M.instability(good) == [], good


def test_pair_list_shows_different_combinations():
    def mol(name, steps):
        r = M.MolResult(name, name + "H2", 0.0, steps[0], steps=steps, forms=[name + "H2"] * len(steps))
        r.reduced, r.mw, r.solubility_mol_l, r.sa = name + "H2", 150.0, 1.0, 2.0
        return r
    star = mol("S", [1.1, 0.02])                 # strong on both sides: pairs with everything
    others = [mol(f"N{i}", [0.05 + 0.01 * i]) for i in range(4)] + [mol(f"P{i}", [0.8 - 0.01 * i]) for i in range(4)]
    import h2screener.pairs as PP
    orig = PP._unstable
    PP._unstable = lambda s: False               # made-up names are not SMILES
    try:
        all_ = PP.best_pairs([star] + others, 20)
        div = PP.best_pairs([star] + others, 20, max_per_molecule=2)
    finally:
        PP._unstable = orig
    count = lambda ps: sum((p.neg.smiles == "S") + (p.pos.smiles == "S") for p in ps)
    assert count(all_) > 2 and count(div) <= 2
    assert len({(p.neg.smiles, p.pos.smiles) for p in div}) == len(div) >= 4


def test_sulfonated_quinones_get_their_hydroquinone_partner():
    assert "O=S(=O)(O)c1ccc2c(O)c3ccccc3c(O)c2c1" in M.reduced_forms("O=C1c2ccccc2C(=O)c2cc(S(=O)(=O)O)ccc21")
    assert "O=S(=O)(O)c1cc(O)c(O)c(S(=O)(=O)O)c1" in M.reduced_forms("O=C1C(=O)C(S(=O)(=O)O)=CC(S(=O)(=O)O)=C1")


def test_proton_shifts_between_heteroatoms_keep_the_carbon_hydrogens():
    # alloxazine reduced forms: protons on different ring nitrogens, same C-H count
    a, b = "O=c1[nH]c2c(c(=O)[nH]1)Nc1ccccc1N2", "O=C1NC(=O)C2=C(N1)Nc1ccccc1N2"
    assert M.carbon_h(a) == M.carbon_h(b)
    # keto-enol: a hydrogen moves onto a carbon
    assert M.carbon_h("Oc1ccc(O)cc1") != M.carbon_h("O=C1CCC(=O)C=C1")


def test_ionised_molecules_do_not_use_esol():
    d = M.descriptors("O=C1c2ccc(S(=O)(=O)O)cc2C(=O)c2cc(S(=O)(=O)O)ccc21")
    assert d["ionised"] and d["sol"] >= M.IONISED_SOL_M
    assert not M.descriptors("O=C1C=CC(=O)C=C1")["ionised"]
