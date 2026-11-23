# h2-screener

Finds and ranks rechargeable cells that use hydrogen as one electrode. It works
from thermodynamic data and physical laws, not from lists of known batteries:
every couple the data allows is generated, then pushed through the same physics.
An AI search (genetic algorithm + machine-learning surrogate + quantum chemistry)
proposes new molecules, and every candidate is sized as a whole battery and
compared with lithium-ion.

## Status and findings

The project is paused and published as is. In short:

- **No candidate beats lithium-ion yet** once sized as a whole battery. The best
  molecule + hydrogen cells reach tens of Wh/kg and cost from about $1,000/kWh
  upwards, against roughly 140 Wh/kg and $100/kWh for a Li-ion pack.
- **The system, not the molecule, is the limit.** Molecules can store a lot of
  energy per kg (several hundred Wh/kg of chemical), but the stack, the platinum,
  the hydrogen storage and the pumps dominate weight and cost.
- **The water limit is a hard wall.** A water-based cell may not charge above
  about 1.63 V (1.229 V + 0.4 V margin). A molecule close to it can only be
  charged at a tiny current, so its stack grows by orders of magnitude.
- **Fast and accurate quantum chemistry disagree outside known chemistry** (see
  below): every promising molecule should be checked with DFT, and in the end
  measured (cyclic voltammetry), before it is trusted.

What may be most useful to others: the screening and system-sizing tool, the
library of computed molecules (several hundred, many never reported in PubChem),
and the xTB / DFT / CCSD(T) error comparison.

## Accuracy: xTB vs DFT

The AI search ranks molecules with GFN2-xTB (seconds per molecule); any molecule
can be recomputed in the background with DFT (wB97X-V/def2-TZVP, 10-30 min).
Errors against measured values (`scripts/validate_abinitio.py`,
[docs/validation_abinitio.md](docs/validation_abinitio.md)):

| Test | GFN2-xTB | DFT wB97X-V | CCSD(T) |
|---|---|---|---|
| Gas reactions, cell voltage, 33 molecules (mean / worst) | 0.62 / 2.1 V | 0.10 / 0.18 V | 0.06 / 0.16 V |
| Energy of formation, test half, 22 molecules (mean) | 106 kJ/mol | 14 kJ/mol | 11 kJ/mol |
| Quinone voltages in water, no fitted constant | 0.4-0.6 V high | 0.02-0.14 V | - |

The search corrects xTB with one constant fitted on known quinones, which makes
it accurate for quinone-like molecules only (test error ~0.01 V on 3 quinones).
On other chemistry the two methods can differ by 0.5-0.9 V: trust DFT, and treat
xTB as a ranking tool.

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

How every part works (simulation, AI search, prices and their sources,
limits): [docs/approach.md](docs/approach.md).

Results that rest on assumptions say so: each input is tagged `database`,
`physics`, `assumption` or `unknown`, and every range is in `targets.toml`.

## Install

Python 3.12, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./setup_accurate.command        # optional: PySCF, for the DFT checks
```

## Visual simulator

```bash
.venv/bin/streamlit run app.py
```

Open http://localhost:8501. In any molecule's details: recompute it with DFT in
the background, switch every number between the xTB and DFT voltage, tag it and
give it a colour (shown on every chart), or add it to your candidates.

Sidebar: what to build (stationary or car pack preset, power, duration,
pressure limits, efficiency target, lifetime, electricity price, ranking
weights). Tabs:

- **Trade-offs**: energy per kg vs build cost (Pareto front, Li-ion lines),
  safety vs cost, energy per kg vs pressure.
- **⭐ Candidates**: first, the best candidates of every battery type in one
  table (table couples, molecule + H2, two molecules, sealed gases), each
  sized as a whole battery, with today's and purpose-built prices and what it
  beats lithium-ion on; the table couples' full detail (pressure, burst energy,
  leak rate, H2 refill interval, chemical drift) is in an expander. Then your
  interesting candidates of every kind
  (couples, molecules, pairs, sealed gas cells), recalculated with the
  settings, with total and per-kWh build price and a one-setting sweep; below,
  the details of any candidate: mass, volume, cost, LCOS, burst energy, leak,
  H2 loss and efficiency against pressure; power vs duration; efficiency vs
  current; mass breakdown; maintenance and closure; every input with its
  provenance; for gas cells, 350 vs 700 bar tanks and where the money goes.
- **AI discovery: 3 battery types** (pick one at the top of the tab):
  molecule + hydrogen gas, two molecules (no gas), and two gases (sealed).
  Each has the same AI loop: genetic generator (SELFIES) + surrogate model +
  GFN2-xTB; the gas search starts from every molecule up to 4 heavy atoms
  and proposes larger ones.
- **Li-ion reference and validation**: Tesla 4680/2170, BYD Blade, pack
  prices (BloombergNEF 2025) with sources, and the model checks.

## Command line

```bash
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

## Author

Nassim MOUALEK (nmoualek@gmail.com)

## License

The code is released under the [MIT License](LICENSE). Third-party data keeps
its own licence: the standard potential table from Wikipedia
(`data/raw/potentials.wikitext` and `data/standard_potentials.csv`, CC BY-SA 4.0)
and ML-HYDPARK (`data/raw/hydpark/`, Witman et al., Sandia National Laboratories,
CC BY 4.0). Sources and revisions are recorded in each `SOURCE.txt`.
