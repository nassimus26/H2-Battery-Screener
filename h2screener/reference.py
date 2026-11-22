"""Reference lithium-ion car batteries (data/reference_batteries.toml) and the
comparison of a sized hydrogen design against them."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load() -> dict:
    with open(ROOT / "data" / "reference_batteries.toml", "rb") as f:
        return tomllib.load(f)


def pack_targets(ref: dict | None = None) -> dict:
    """The figures a hydrogen design has to beat: the best pack-level values."""
    ref = ref or load()
    packs = [b for b in ref["battery"] if b["level"] == "pack"]
    wh_kg = max(max(b.get("wh_kg", [0])) for b in packs)
    wh_l = max(max(b.get("wh_l", [0])) for b in packs)
    charge = min(min(b.get("charge_minutes_10_80", [math.inf])) for b in packs)
    p = ref["prices"]
    return {"wh_kg": wh_kg, "wh_l": wh_l, "charge_min_10_80": charge,
            "usd_kwh_lfp": p["pack_lfp_usd_kwh"], "usd_kwh_stationary": p["pack_stationary_usd_kwh"],
            "usd_kwh_bev": p["pack_bev_usd_kwh"], "cycles_lfp": 3000}


def compare(point, e_kwh: float, ref: dict | None = None) -> dict[str, str]:
    """better / worse / unknown for each figure of a sized DesignPoint holding
    e_kwh of delivered energy."""
    t = pack_targets(ref)
    wh_kg = e_kwh * 1000 / point.mass_kg if point.mass_kg else 0
    wh_l = e_kwh * 1000 / point.volume_l if point.volume_l else 0
    charge_min = point.min_charge_h * 60 * 0.7     # 10-80 % is 70 % of the capacity
    return {
        "energy per kg": "better" if wh_kg > t["wh_kg"] else "worse",
        "energy per litre": "better" if wh_l > t["wh_l"] else "worse",
        "build cost": "better" if point.capex_usd_kwh < t["usd_kwh_stationary"] else "worse",
        "charge 10-80 %": "better" if charge_min < t["charge_min_10_80"] else "worse",
        "lifetime": "unknown",
    }
