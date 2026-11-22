"""Molecule production-cost estimate."""

import pytest

from h2screener import synthesis as S
from h2screener.explain import explain_cost
from h2screener.reactions import Species


def test_the_fit_reproduces_the_priced_examples():
    """Every anchor within x4 of its price, most within the stated x3 range
    (the difficulty score tracks price only loosely)."""
    cfg = S.default_config()
    inside = 0
    for a in cfg["anchors"]:
        e = S.estimate(a["smiles"], cfg)
        assert e.usd_kg / 4 <= a["usd_kg"] <= e.usd_kg * 4, a["name"]
        inside += e.lo <= a["usd_kg"] <= e.hi
    assert inside >= len(cfg["anchors"]) - 1


def test_harder_molecules_cost_more():
    easy = S.estimate("Oc1ccc(O)cc1")                       # hydroquinone
    hard = S.estimate("OC1C2CC3(O)CC1(C=O)C3N2")            # bridged, many stereocentres
    assert hard.difficulty > easy.difficulty + 3
    assert hard.usd_kg > 30 * easy.usd_kg
    assert hard.extrapolated and not easy.extrapolated


def test_price_factor_scales_the_estimate():
    cfg = dict(S.default_config(), price_factor=3.0)
    assert S.estimate("Oc1ccc(O)cc1", cfg).usd_kg == pytest.approx(3 * S.estimate("Oc1ccc(O)cc1").usd_kg)


def test_species_price_uses_synthesis_for_molecules_only():
    mol = Species("C6H4O2", 0, "aq", "O=C1C=CC(=O)C=C1 (positive)")
    price, sc = S.species_price(mol, 50.0)
    assert sc is not None and sc.smiles == "O=C1C=CC(=O)C=C1" and price == sc.usd_kg
    oxide = Species("MnO2", 0, "s", "MnO2")
    price, sc = S.species_price(oxide, 50.0)
    assert sc is None and 0 < price < 50


def test_search_penalty_spares_cheap_molecules():
    assert S.price_penalty(5.0) == 1.0
    assert S.default_config()["search_min_factor"] <= S.price_penalty(1000.0) < 0.5


def test_explanation_names_the_molecule_price_and_the_stack_range():
    sc = S.estimate("O=C1C=CC(=O)C=C1")
    lines = explain_cost({"cell stack": 320000, "chemicals": 400},
                         {"e_kwh": 60, "power_kw": 200, "i_a_cm2": 0.034, "area_m2": 1068, "v_dis": 0.55,
                          "synth": [(sc, 80.0)]})
    text = " ".join(lines)
    assert "Making the molecule O=C1C=CC(=O)C=C1" in text and "difficulty" in text
    assert "300 USD." not in text
