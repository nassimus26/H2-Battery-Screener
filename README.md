# h2-screener

Finds and ranks rechargeable cells that use hydrogen as one electrode. It works
from thermodynamic data and physical laws, not from lists of known batteries:
every couple the data allows is generated, then pushed through the same physics.

## What it does

| Stage | Method | State |
|---|---|---|
| 1. Generate | every half-reaction of the standard potential table x every hydrogen electrode (acid, alkaline); every `A + x H2 -> B + y H2O` and `M + x H2 -> MH2x` between Materials Project phases, including computed, never-made ones | table: works offline; MP: needs `MP_API_KEY` (code written, not yet run) |
| 2. Voltage, energy | E0, Nernst term with real H2 fugacity, Wh/kg of reactants | works |
| 3. Real gas | CoolProp equations of state: density, fugacity, condensation (Cl2 is liquid above ~7 bar) | works |
| 4. Stability | water oxidation above 1.229 V + margin; radioactive elements excluded; products that are very hard to turn back (N2, CO2, ...) flagged | works (Pourbaix check through MP: to do) |
| 5. New compounds | universal ML interatomic potentials (CHGNet / MACE-MP) on ion-substituted solids | to do (molecules: see below) |
| 6. Safety, cost | element prices and supply concentration (pymatgen data), PubChem GHS hazards (`--ghs`) | works |
| 7. Efficiency, power, recharge | Butler-Volmer kinetics, ohmic gap/conductivity, Fick's law with Stokes-Einstein diffusion and Henry's-law gas supply; Monte Carlo over the unknown inputs, p10/p50/p90, and the input that dominates | works |
| 7b. Weight vs pressure | gas swing volume from real-gas density, thin-wall vessel mass `phi rho p V / sigma`, stack, catalyst; sweep of maximum pressure | works |
| 8. Lifetime | no law predicts it | reported as unknown |
| Safety vs pressure | isentropic burst energy (kg TNT), choked-flow leak, time to 25 % of the H2 flammability limit, p x V | works |
| Maintenance | H2 permeation (316L Arrhenius fits from h-transport-materials, Sieverts' law) and seal leaks; chemical closure from measured and derived (Latimer) half-reactions: over-reduction, disproportionation, O2 recombination | works |
| Design | power x duration sizing, current density chosen for the lowest levelised cost, Pareto front | works |
| Molecules | any molecule: H2-addition partners, xTB/ALPB voltages calibrated on p-benzoquinone (+/-0.3 V), tautomer check, ESOL solubility, SA score, PubChem novelty; AI search loop | works |

Results that rest on assumptions say so: each input is tagged `database`,
`physics`, `assumption` or `unknown`, and every range is in `targets.toml`.

## Visual simulator

```bash
.venv/bin/streamlit run app.py
```

Sidebar: what to build (stationary or car pack preset, power, duration,
pressure limits, efficiency target, lifetime, electricity price, ranking
weights). Tabs:

- **Best candidates**: every sized couple with system Wh/kg and Wh/L, build
  cost, levelised cost of storage, efficiency band, fastest charge, burst
  energy, leak rate, H2 refill interval, chemical drift, and better/worse
  against current car lithium-ion packs.
- **Trade-offs**: energy per kg vs build cost (Pareto front, Li-ion lines),
  safety vs cost, energy per kg vs pressure.
- **Candidate detail**: mass, volume, cost, LCOS, burst energy, leak, H2 loss
  and efficiency against pressure; power vs duration; efficiency vs current;
  mass breakdown; maintenance and closure; every input with its provenance.
- **AI molecule discovery**: genetic generator (SELFIES) + surrogate model +
  GFN2-xTB voltages; marks molecules absent from PubChem; sizes chosen
  molecules as batteries.
- **Li-ion reference and validation**: Tesla 4680/2170, BYD Blade, pack
  prices (BloombergNEF 2025) with sources, and the model checks.

## Command line

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m h2screener run --top 30          # out/report.md, out/candidates.csv
MP_API_KEY=... .venv/bin/python -m h2screener run --mp all
.venv/bin/python -m h2screener validate             # model priors vs published cells
.venv/bin/python -m pytest -q tests
```

`data/standard_potentials.csv` is rebuilt from the downloaded table with
`scripts/parse_potentials.py`; the source page and revision are recorded there.

## Validation

Published hydrogen cells are never candidates and are not used for fitting.
They check whether the physics and the assumed ranges cover reality:

- Mn-H2 (Chen et al., Nature Energy 3 (2018) 428), ~1.3 V measured: model OCV 1.22 V (inside the check).
- H2-Br2 (Cho et al., J. Electrochem. Soc. 159 (2012) A1806), 1.4 W/cm2 peak measured:
  model p90 is 0.72 W/cm2 (outside). The generic range for the reaction speed of
  an untested electrode is too pessimistic for fast halogen couples. This is the
  input a kinetics model (stage 5 descriptors / ML) has to supply.

## Next

1. Materials Project generator (`--mp`), then MP Pourbaix stability for every candidate.
2. Per-couple kinetics estimates instead of one wide prior: Marcus reorganisation
   energies from xTB for molecular couples, and a regression of measured exchange
   currents on element/structure descriptors, with symbolic regression (PySR) to
   look for simple predictive equations. Found equations are hypotheses until
   they predict held-out measurements.
3. New compounds: substitution of ions in known hydrogen hosts, relaxed with a
   universal ML potential, kept when within 50 meV/atom of the MP hull.
4. Active learning: choose the next candidate to compute where the expected
   gain in score is largest.
# bgaz
# bgaz
