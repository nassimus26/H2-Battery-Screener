# Sealed two-gas batteries (tab "AI discovery", type "Two gases (sealed)")

Goal: a battery that stores its energy in two gases, so it can be much lighter than Li-ion.
Code: `h2screener/gases.py` (generation, energies, couples, cells), `options.gas_configurations`
(sizing), tests in `tests/test_gases.py`.

## What it does

1. **Generates every molecule** with up to 4 atoms of C, N, O, F, S, Cl: about 2,000 molecules,
   including radicals and diatomics from the measured table. Molecules that are gases at room temperature are
   small, so the whole space is computed and nothing is sampled.
2. **Boiling point:** measured where known, else the Joback estimate (±25 K).
   A gas that condenses below the tank pressure is kept as a liquid at its vapour pressure, which makes
   a lighter tank.
3. **Energy of each molecule** (Gibbs energy of formation at 25 °C):
   - measured (NIST-JANAF / CRC) for about 45 known molecules;
   - otherwise GFN2-xTB in the gas phase, converted with atom equivalents fitted to the measured
     molecules. The leave-one-out error of that fit is shown in the tab and used as the voltage
     uncertainty;
   - molecules that would not survive storage are removed, using the same instability alerts as the
     liquid search plus strained rings, carbenes and charge-separated forms.
4. **Couples:** `kA A + a H2 → kB B + w H2O(l)` between molecules with the same C, N, S, F, Cl, for example
   O2/H2O, CO2/CH4, N2/NH3, Cl2/HCl. The voltage is −ΔG/(2aF) against the hydrogen electrode.
5. **Cells:** an oxidant couple (charged form a gas) against a fuel couple (charged form a gas;
   hydrogen is the simplest), with an acid membrane. About 140,000 cells.
6. **Ranking:** energy per mole of stored gas. A tank wall weighs the same per mole of gas at any
   pressure, so this number sets how light the battery can be. The best cells are then sized in
   full for the sidebar settings: carbon-fibre tanks at 350 and 700 bar, cells, platinum, power
   electronics, efficiency, and the cost range over reaction speed.
7. **Evidence:** each couple says whether it is proven rechargeable in a cell (H2, O2/H2O, Cl2/HCl),
   works in one direction only (CH4, CO, methanol, NH3...), or is untested.

## AI search for larger gas molecules (`h2screener/gas_search.py`)

The enumeration stops at 4 heavy atoms, but some larger molecules are still gases (C5H4, neopentane,
fluorinated chains) or useful liquid products. The same loop as the liquid search proposes them:

1. **Pool:** every molecule up to 4 heavy atoms (all computed) plus every molecule the search has computed.
   The simple molecules are never dropped: they are the parents and the partners of every new molecule.
2. **Generate:** SELFIES mutations and crossovers of the molecules in the lightest cells (C, N, O, F, S, Cl).
   Everything up to 4 heavy atoms is known already, so children are the larger ones.
3. **Filter:** neutral, closed shell, one fragment, the storage alerts of the screen, boiling below 400 K (Joback).
4. **Predict:** extra-trees surrogate on Morgan fingerprints and atom/bond counts, trained on every xTB energy,
   predicts the Gibbs energy of formation per atom, with the spread of the trees as its uncertainty.
5. **Acquire:** each child is built into cells against the 30 best oxidant and fuel couples of the pool at the
   predicted energy ± kappa × spread; the batch is the highest upper bounds on the lightest cell (Wh/kg,
   chemicals + tank walls).
6. **Compute:** GFN2-xTB for the batch, then the true lightest cell; retrain and repeat.

Results go to `cache/gas_search.json`; their energies go to the same xTB cache, and they join the full screen
and the sizing table (column "found by the AI").

Couples now allow up to 8 molecules of one species and 60 electrons (`MAX_K`, `MAX_E`; before: 3 and 12),
so a C5–C8 fuel can be oxidised all the way to CO2. This also adds the full oxidation of propane and butane
to the small-molecule screen: 292,607 cells instead of 172,943.

Limit: xTB with atom equivalents carries about ±80 kJ/mol, more for larger molecules, and keeping the best
of many uncertain values favours molecules whose energy is overestimated. Check a winner with a better method.

## Limits

- Whether a reaction runs backwards cleanly cannot be computed. This is the main risk of every
  candidate other than H2/O2.
- Cells above the water window (fluorine, OF2, about 2 V) cannot be charged in a water electrolyte and are
  dropped by the sizing.
- Products that dissolve in the electrolyte (HCl, HF, NH3) are treated as pure gases or liquids. Dissolving
  changes their voltage by tenths of a volt: the real H2/Cl2 cell is 1.36 V, the gas-phase value 0.99 V.
- The xTB energies of small inorganic molecules are rough (N2O, HF, NF3 are off by about 100 kJ/mol in the fit).
  Molecules marked "computed only" may not exist.
- Not counted: pumps, cooling, the water tank wall, a cell housing built for pressure, and the safety of
  storing high-pressure oxygen next to a fuel.

## Results (2026-11-22, full run)

- 1,687 molecules computed with xTB; after the stability filter, 445 molecules have an energy and 134 of them are
  gases at 25 °C. That gives 172,943 cells that can be charged in water.
- Atom-equivalent fit: 41 measured molecules, leave-one-out error 81 kJ/mol (about 0.4 V on a 2-electron
  step). Voltages of "computed only" molecules are rough; measured ones are good to about 0.01 V.
- 4,706 cells store more energy per mole of gas than H2 + O2, 113 of them using measured molecules only (mostly
  hydrocarbons, ethylene oxide or ethanol burnt with O2: 8–14 electrons per fuel molecule).
- **After full sizing, H2 + O2 is still the lightest and the only proven rechargeable one.** The fuel gases react slowly
  (unknown kinetics, sampled), so they need bigger stacks and run at lower efficiency.

| setting | best cell | Wh/kg | Wh/L | build $/kWh | efficiency |
|---|---|---|---|---|---|
| car 220 kW × 0.38 h | O2 + 2 H2 → 2 H2O, carbon fibre 350 bar | **503** | 252 | 451 | 0.43 |
| car | 5 O2 + 2 C2H2 (acetylene) → 4 CO2 + 2 H2O | 307 | 174 | 677 | 0.30 |
| stationary 100 kW × 6 h | O2 + 2 H2 → 2 H2O | **764** | 382 | 119 | 0.61 |
| stationary | acetylene + O2 | 682 | 422 | 143 | 0.45 |
| stationary | ethanol + O2 → acetic acid + water | 616 | 493 | 122 | 0.47 |
| Li-ion pack | | 140 | 208 | 70 | ~0.9 |

Several hydrocarbon cells are denser per litre than H2 + O2: their products (CO2, acids) condense. No one has shown
any of them running backwards in a cell.
