"""Plain-language cost explanations."""

from h2screener import stackcost
from h2screener.explain import explain_cost


def test_power_heavy_design_points_to_duration_and_power():
    lines = explain_cost({"cell stack": 9000, "power electronics": 2000, "chemicals": 500},
                         {"e_kwh": 60, "power_kw": 200, "i_a_cm2": 0.01, "area_m2": 900})
    text = " ".join(lines)
    assert "cell stack" in lines[0] and "Delivers full power for" in text and "Round-trip efficiency" in text


def test_chemical_heavy_design_points_to_molecule_size():
    lines = explain_cost({"chemicals": 9000, "cell stack": 100}, {"e_kwh": 600, "power_kw": 100,
                                                                 "chem_kg_per_kwh": 12.0, "chem_usd_kg": 3.0})
    assert any("Largest molecule" in x for x in lines)


def test_tank_heavy_design_points_to_pressure():
    lines = explain_cost({"hydrogen tank": 6000, "cell stack": 1000}, {"e_kwh": 600, "power_kw": 100, "p_max_bar": 3})
    assert any("pressure" in x for x in lines)


def test_stack_cost_shows_the_arithmetic():
    lines = explain_cost({"cell stack": 320000, "power electronics": 20000},
                         {"e_kwh": 60, "power_kw": 200, "i_a_cm2": 0.034, "area_m2": 1068, "v_dis": 0.55,
                          "stack_est": stackcost.stack("liquid", "mass")})
    text = " ".join(lines)
    assert "1,068 m2" in text and "187 W" in text and "160,200 USD" in text
    assert "Noack" in text and "1,602-4,450 USD/kWh" in text and "number of m2" in text


def test_stack_prices_follow_cell_kind_and_scale():
    g, l = stackcost.stack("gas", "mass"), stackcost.stack("liquid", "mass")
    assert l.usd < g.usd < stackcost.stack("gas", "small").usd
    assert stackcost.stack("liquid", "small").usd > 10 * l.usd      # prototypes cost far more
    for e in (g, l, stackcost.POWER_ELECTRONICS):
        assert e.lo < e.usd < e.hi and e.basis
