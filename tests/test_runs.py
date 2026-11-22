"""Saved settings and saved searches."""

from types import SimpleNamespace

from h2screener import runs
from h2screener import molecules as M


def test_run_record_keeps_settings_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    rid = runs.new_run("pair", {"cfg_power": 100.0}, {"generations": 3, "batch": 4, "max_heavy": 16, "kappa": 1.5})
    r = M.MolResult("O=C1C=CC(=O)C=C1", "Oc1ccc(O)cc1", 1.5, 0.7, steps=[0.7], forms=["Oc1ccc(O)cc1"])
    ctl = SimpleNamespace(status="finished", history=[{"generation": 1, "best": 12.0, "molecules": 1}],
                          evaluated=[r.smiles])
    runs.update_run(rid, ctl, {r.smiles: r})
    rec = runs.list_runs("pair")[0]
    assert rec["settings"]["cfg_power"] == 100.0 and rec["status"] == "finished" and rec["ended"]
    assert M.MolResult(**rec["molecules"][0]).steps == [0.7]
    assert "3 gen x 4" in runs.label(rec)


def test_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "CACHE", tmp_path)
    monkeypatch.setattr(runs, "SETTINGS", tmp_path / "s.json")
    runs.save_settings({"cfg_pmax": 45})
    assert runs.load_settings() == {"cfg_pmax": 45}


def test_delete_moves_to_trash(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    rid = runs.new_run("positive", {}, {})
    assert runs.delete_run(rid)
    assert runs.list_runs() == [] and (tmp_path / "runs" / "trash" / f"{rid}.json").exists()


def test_remove_results_keeps_shared_molecules(tmp_path, monkeypatch):
    from dataclasses import asdict
    from h2screener import ai_search
    monkeypatch.setattr(runs, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(ai_search, "STORE", tmp_path / "store.csv")
    a = M.MolResult("O=C1C=CC(=O)C=C1", "Oc1ccc(O)cc1", 1.5, 0.7, steps=[0.7], forms=["Oc1ccc(O)cc1"])
    b = M.MolResult("O=C1C=CC(=O)c2ccccc21", "Oc1ccc(O)c2ccccc12", 1.1, 0.33, steps=[0.33], forms=["x"])
    ai_search.save_store({a.smiles: a, b.smiles: b})
    r1 = runs.new_run("pair", {}, {})
    rec1 = {"id": r1, "molecules": [asdict(a), asdict(b)]}
    other = runs.new_run("positive", {}, {})
    import json
    p = runs.RUNS / f"{other}.json"
    d = json.loads(p.read_text()); d["molecules"] = [asdict(a)]; p.write_text(json.dumps(d))
    assert runs.remove_results(rec1) == 1            # b removed, a kept (shared)
    assert set(ai_search.load_store()) == {a.smiles}
