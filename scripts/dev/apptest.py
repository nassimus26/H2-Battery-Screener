import faulthandler; faulthandler.dump_traceback_later(240, repeat=False)
import sys, time, pathlib, tempfile
sys.path.insert(0, ".")
from h2screener import runs, ai_search, gas_search
_tmp = pathlib.Path(tempfile.mkdtemp())
runs.SETTINGS = _tmp / "ui_settings.json"   # never touch the user's settings
runs.RUNS = _tmp / "runs"                   # nor the user's saved searches (the app marks running ones as interrupted at start)
# never touch the user's AI results either
ai_search.STORE.parent  # noqa
from streamlit.testing.v1 import AppTest
APP = str(pathlib.Path.cwd() / "app.py")
SECTIONS = ["🧪 AI discovery: 3 battery types", "⭐ Candidates (best + interesting + details)", "Trade-offs",
            "Li-ion reference and validation"]
at = AppTest.from_file(APP, default_timeout=900)
for sec in SECTIONS:
    for kind in (["molecule", "pair", "gas"] if sec.startswith("🧪") else [None]):
        at.session_state["cfg_tab"] = sec
        if kind:
            at.session_state["cfg_kind"] = kind
        t0 = time.time(); at.run()
        print(f"{sec[:20]:20s} {kind or '':8s} {time.time() - t0:5.0f} s exceptions:", [e.value for e in at.exception][:2])
at.session_state["cfg_tab"] = SECTIONS[1]
for src in ("couple", "molecule", "pair", "gascell"):
    at.session_state["cand_src"] = src
    t0 = time.time(); at.run()
    print(f"candidates/{src:9s} {time.time() - t0:5.0f} s exceptions:", [e.value for e in at.exception][:2])
for kind, ids in (("gascell", ["O=O>O|H2"]), ("molecule", ["N#CCCNCCCNCCNCCNC1N=c2cnnnc2=N1"])):
    at.session_state["pending_dialog"] = (kind, ids)
    t0 = time.time(); at.run()
    print(f"dialog {kind:9s} {time.time() - t0:5.0f} s exceptions:", [e.value for e in at.exception][:2])
# reset: stop + clear caches only (never the data in a test)
at.session_state["rs_data"] = False
at.session_state["rs_settings"] = False
btn = [b for b in at.button if b.key == "rs_go"]
print("reset button found:", bool(btn))
if btn:
    btn[0].click(); t0 = time.time(); at.run()
    print(f"after reset {time.time() - t0:5.0f} s exceptions:", [e.value for e in at.exception][:2],
          "| toasts:", [x.value for x in at.toast][:1])
# ⭐ candidates numbered on the trade-off chart and in the tables
at6 = AppTest.from_file(APP, default_timeout=900)
at6.session_state["watchlist"] = [["gascell", "O=O>O|H2"], ["couple", "x"]]
at6.session_state["cfg_tab"] = "Trade-offs"
t0 = time.time(); at6.run()
print(f"pins on chart {time.time() - t0:4.0f} s exceptions:", [e.value for e in at6.exception][:2])
at6.session_state["cfg_tab"] = SECTIONS[1]
at6.run()
print("pins in tables exceptions:", [e.value for e in at6.exception][:2])
for d in at6.dataframe:
    v = d.value
    if "candidate" in v.columns:
        print([c for c in v["candidate"].astype(str) if c.startswith("⭐")][:3])
# show switch: only AI finds + pins
at6.session_state["cfg_tab"] = "Trade-offs"
at6.session_state["trade_show"] = "ai"
at6.run()
print("only-AI view exceptions:", [e.value for e in at6.exception][:2], "| infos:", [i.value[:60] for i in at6.info][:2])
# known/new labels and the assessment
at7 = AppTest.from_file(APP, default_timeout=900)
for kind, ids in (("gascell", ["O=O>O|H2"]), ("molecule", ["N#CCCNCCCNCCNCCNC1N=c2cnnnc2=N1"])):
    at7.session_state["pending_dialog"] = (kind, ids)
    t0 = time.time(); at7.run()
    md = [m.value for m in at7.markdown if "Is it interesting" in m.value or m.value.startswith("- **Against")]
    print(f"assess {kind} {time.time() - t0:4.0f} s exceptions:", [e.value for e in at7.exception][:2],
          "| banners:", [x.value[:70] for x in list(at7.success) + list(at7.info)][:2], "| has assessment:", bool(md))
