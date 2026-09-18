from __future__ import annotations
import csv, json
from pathlib import Path
import pytest
from tools import hard_cases

def test_generator_is_deterministic_and_blind(tmp_path: Path):
    a,b=tmp_path/"a",tmp_path/"b"; common=["generate","--count","6","--absent","2","--seed","41"]
    assert hard_cases.main([*common,"--output",str(a)])==0
    assert hard_cases.main([*common,"--output",str(b)])==0
    assert (a/"manifest.csv").read_bytes()==(b/"manifest.csv").read_bytes()
    with (a/"pairs.csv").open(newline="",encoding="utf-8") as f: pairs=list(csv.DictReader(f))
    assert len(pairs)==6 and tuple(pairs[0])==("pair_id","reference_path","search_path")
    with (a/"manifest.csv").open(newline="",encoding="utf-8") as f: manifest=list(csv.DictReader(f))
    assert all(float(r["uniqueness_margin"])>=.03 for r in manifest if r["present"]=="1")
    assert all(float(r["intensity_error_px"])<=2 for r in manifest if r["present"]=="1")
    assert all(float(r["edge_error_px"])<=5 for r in manifest if r["present"]=="1")
    assert all(r["no_match_confirmed"]=="1" for r in manifest if r["present"]=="0")

def test_perfect_predictions_score_85(tmp_path: Path):
    suite=tmp_path/"suite"; hard_cases.main(["generate","--output",str(suite),"--count","4","--absent","1","--seed","7"])
    with (suite/"ground_truth.csv").open(newline="",encoding="utf-8") as f: truth=list(csv.DictReader(f))
    predictions=tmp_path/"predictions.csv"
    with predictions.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=hard_cases.PRED); w.writeheader()
        for r in truth:
            found=int(r["present"]); w.writerow({"pair_id":r["pair_id"],"x":r["x"],"y":r["y"],"theta":r["theta"],"scale":r["scale"],"found":found,"score":.99 if found else .01})
    out,fail=tmp_path/"score.json",tmp_path/"failures.csv"
    hard_cases.main(["score","--truth",str(suite/"ground_truth.csv"),"--predictions",str(predictions),"--output",str(out),"--failures",str(fail)])
    report=json.loads(out.read_text(encoding="utf-8")); assert report["measured_total_85"]==85.; assert report["failure_count"]==0

def test_scorer_rejects_non_contract_prediction(tmp_path: Path):
    truth=tmp_path/"truth.csv"; predictions=tmp_path/"predictions.csv"
    truth.write_text("pair_id,present,x,y,theta,scale\np,0,0,0,0,0\n",encoding="utf-8")
    predictions.write_text("pair_id,x,y,theta,scale,found,score\np,1,0,0,0,0,1.1\n",encoding="utf-8")
    with pytest.raises(ValueError,match="score outside"):
        hard_cases.main(["score","--truth",str(truth),"--predictions",str(predictions),"--output",str(tmp_path/"score.json")])

