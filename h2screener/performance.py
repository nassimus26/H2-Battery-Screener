"""Stage 7: cell performance from physical laws - efficiency, power, recharge.

Cell voltage at current density i (A/cm2) and state of charge s:

  V_dis = OCV(s) - eta_act,pos - eta_act,neg - i*ASR - eta_mt
  V_ch  = OCV(s) + eta_act,pos + eta_act,neg + i*ASR + eta_mt

  OCV(s)   Nernst: E0 + sum_gas (nu RT/nF) ln(f_gas(p(s))/p0)
                   + (RT/nF) ln(s/(1-s)) when the positive side is dissolved
  eta_act  Butler-Volmer, alpha = 1/2: (2RT/F) asinh(i / 2 i0)
  ASR      gap / conductivity + separator and contact resistance
  eta_mt   -(RT/nF) ln(1 - i/i_lim), i_lim = n F D_eff c / delta
           (Fick's law; D from Stokes-Einstein for ions, Bruggeman-corrected
           gas diffusion for gas-fed electrodes, c from the real-gas density)

Inputs that no law fixes (exchange currents of new couples, solubility,
separator resistance) are sampled from wide ranges, so every output is a
distribution (p10/p50/p90), and the report says which input dominates it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import realgas
from .constants import F, P_STD, R
from .provenance import Value
from .reactions import CellReaction

K_B = 1.380649e-23
N_A = 6.02214076e23


def water_viscosity(t_k: float) -> float:
    """Pa s, Vogel equation fit for liquid water (valid 273-373 K)."""
    return 1e-3 * math.exp(-3.7188 + 578.919 / (t_k - 137.546))


def stokes_einstein(t_k: float, radius_m: float) -> float:
    """m2/s"""
    return K_B * t_k / (6 * math.pi * water_viscosity(t_k) * radius_m)


@dataclass
class CellInputs:
    """Sampled arrays (one entry per Monte Carlo draw)."""
    i0_pos: np.ndarray
    i0_neg: np.ndarray
    asr: np.ndarray
    ilim: np.ndarray
    ce: np.ndarray


@dataclass
class CellModel:
    cell: CellReaction
    t_k: float
    p_min: float
    p_max: float
    params: dict[str, Value]
    fixed_fugacity: dict = None

    def __post_init__(self):
        self.fixed_fugacity = self.fixed_fugacity or {}
        self.gas = {g: nu for g, nu in self.cell.gas_terms().items() if realgas.fluid(g)}
        self.dissolved = self.cell.dissolved_moles() > 0
        self.rt_nf = R * self.t_k / (self.cell.n * F)
        # H2 density at the ends of the pressure swing: s=1 full (p_max).
        if "H2" in self.gas:
            self.d_lo = realgas.molar_density("H2", self.t_k, self.p_min)
            self.d_hi = realgas.molar_density("H2", self.t_k, self.p_max)

    def pressure(self, s: np.ndarray) -> np.ndarray:
        if "H2" not in self.gas:
            return np.full_like(s, self.p_max)
        d = self.d_lo + s * (self.d_hi - self.d_lo)
        return np.array([realgas.pressure_at_density("H2", self.t_k, x) for x in d])

    def ocv(self, s: np.ndarray) -> np.ndarray:
        v = np.full_like(s, self.cell.e0, dtype=float)
        p = self.pressure(s)
        for g, nu in self.gas.items():
            if g in self.fixed_fugacity:          # e.g. O2 from air
                f = np.full_like(s, self.fixed_fugacity[g])
            elif g == "H2":
                f = np.array([realgas.fugacity("H2", self.t_k, x) for x in p])
            else:  # other stored gases are kept at p_max on their own side
                f = np.full_like(s, realgas.fugacity(g, self.t_k, self.p_max))
            v += nu * self.rt_nf * np.log(f / P_STD)
        if self.dissolved:
            # both electrolytes change concentration in a two-molecule cell
            sides = 2 if self.cell.origin == "two-molecule" else 1
            sc = np.clip(s, 1e-3, 1 - 1e-3)
            v += sides * self.rt_nf * np.log(sc / (1 - sc))
        return v

    def overpotential(self, i: np.ndarray, x: CellInputs) -> np.ndarray:
        rt_f = R * self.t_k / F
        act = 2 * rt_f * (np.arcsinh(i / (2 * x.i0_pos)) + np.arcsinh(i / (2 * x.i0_neg)))
        frac = np.clip(i / x.ilim, 0, 1 - 1e-9)
        mt = -self.rt_nf * np.log(1 - frac)
        return act + i * x.asr + mt

    def sample(self, rng: np.random.Generator, n: int) -> CellInputs:
        p = self.params
        return CellInputs(
            p["i0_pos"].sample(rng, n), p["i0_neg"].sample(rng, n),
            p["asr"].sample(rng, n), p["ilim"].sample(rng, n), p["ce"].sample(rng, n),
        )


SOC = np.linspace(0.1, 0.9, 9)


def efficiency(model: CellModel, x: CellInputs, i: np.ndarray, ocv: np.ndarray):
    """Round-trip energy efficiency at symmetric current density i (per draw).
    ocv: OCV on the SOC grid. Returns (rte, mean V_dis, max V_ch)."""
    eta = model.overpotential(i, x)[:, None]
    vd = ocv[None, :] - eta
    vc = ocv[None, :] + eta
    vd_mean = np.clip(vd, 0, None).mean(axis=1)
    rte = x.ce * vd_mean / vc.mean(axis=1)
    rte = np.where((vd > 0).all(axis=1), rte, 0.0)
    return rte, vd_mean, vc.max(axis=1)


def evaluate(model: CellModel, rte_target: float, v_ch_max: float, q_area_ah: Value,
             rng: np.random.Generator, n: int = 400) -> dict:
    """Monte Carlo over the uncertain inputs. Returns percentile summaries."""
    x = model.sample(rng, n)
    ocv = model.ocv(SOC)
    # Largest charge current with RTE >= target and V_ch <= v_ch_max (bisection, per draw).
    lo = np.full(n, 1e-6)
    hi = x.ilim * 0.999
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        rte, _, vmax = efficiency(model, x, mid, ocv)
        ok = (rte >= rte_target) & (vmax <= v_ch_max)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    i_ok = lo
    rte_ok, vd_ok, _ = efficiency(model, x, i_ok, ocv)
    feasible = rte_ok >= rte_target - 1e-6
    i_ok = np.where(feasible, i_ok, 0.0)
    # Peak power at mid charge.
    grid = np.geomspace(1e-4, 1, 60)[None, :] * x.ilim[:, None]
    ocv_mid = model.ocv(np.array([0.5]))[0]
    eta = np.stack([model.overpotential(grid[:, k], x) for k in range(grid.shape[1])], axis=1)
    p_peak = (grid * np.clip(ocv_mid - eta, 0, None)).max(axis=1)
    q = q_area_ah.sample(rng, n)
    t_charge_h = np.where(i_ok > 0, q / np.maximum(i_ok, 1e-12), np.inf)
    # Efficiency at the current that charges the design capacity in a fixed time is
    # reported separately by the system stage; here: sensitivity of i_ok to inputs.
    sens = {}
    logi = np.log(np.maximum(i_ok, 1e-9))
    for k, arr in (("i0_pos", x.i0_pos), ("i0_neg", x.i0_neg), ("asr", x.asr),
                   ("ilim", x.ilim), ("ce", x.ce)):
        if np.ptp(arr) > 0 and np.ptp(logi) > 0:
            sens[k] = float(abs(np.corrcoef(np.log(arr), logi)[0, 1]))
    return {
        "ocv_v": (float(ocv.min()), float(ocv.mean()), float(ocv.max())),
        "feasible_frac": float(feasible.mean()),
        "i_charge_a_cm2": pct(i_ok),
        "rte_at_i_charge": pct(np.where(feasible, rte_ok, 0)),
        "v_dis_avg": pct(np.where(feasible, vd_ok, 0)),
        "p_at_target_w_cm2": pct(i_ok * vd_ok),
        "p_peak_w_cm2": pct(p_peak),
        "t_charge_h": pct(t_charge_h),
        "dominant_unknown": max(sens, key=sens.get) if sens else "",
    }


def pct(a: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.percentile(a, [10, 50, 90]))


def efficiency_at(model: CellModel, i: float, rng: np.random.Generator, n: int = 400):
    """RTE distribution at a given symmetric current density (validation use)."""
    x = model.sample(rng, n)
    rte, vd, _ = efficiency(model, x, np.full(n, i), model.ocv(SOC))
    return pct(rte), pct(vd)


def peak_power(model: CellModel, rng: np.random.Generator, n: int = 400):
    x = model.sample(rng, n)
    grid = np.geomspace(1e-4, 1, 80)[None, :] * x.ilim[:, None]
    ocv_mid = model.ocv(np.array([0.5]))[0]
    eta = np.stack([model.overpotential(grid[:, k], x) for k in range(grid.shape[1])], axis=1)
    return pct((grid * np.clip(ocv_mid - eta, 0, None)).max(axis=1))
