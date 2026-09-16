"""Verify packaged source and recorded evidence without ML dependencies."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT/path).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    sources = [ROOT / f for f in ["demo.py","evaluate.py","explain_logreg.py","explain_tinybert.py"]]
    sources += list((ROOT/"src").glob("*.py")) + list((ROOT/"scripts").glob("*.py"))
    for path in sources:
        ast.parse(path.read_text(),filename=str(path))

    manifest = load("data/split_manifest.json")
    for relative,digest in manifest["sha256"].items():
        path = ROOT/relative
        if path.exists():
            check(hashlib.sha256(path.read_bytes()).hexdigest()==digest,f"CSV changed: {relative}")
        else:
            check(relative=="data/train.csv",f"Missing required sample: {relative}")
    rows = list(csv.DictReader((ROOT/"data/eval_clean.csv").open()))
    candidates = list(csv.DictReader((ROOT/"data/sample_test.csv").open()))
    check(len(rows)==manifest["evaluation_rows"]==39,"Evaluation size mismatch")
    check(len({r["text"] for r in rows})==len(rows),"Duplicate evaluation texts")
    check(all(r in candidates for r in rows),"Evaluation row absent from candidate sample")
    check(all(r["text"].strip() and r["label"] in ("0","1") for r in rows),"Invalid evaluation row")
    pos = sum(r["label"]=="1" for r in rows)
    neg = len(rows)-pos
    check((neg,pos)==(20,19),"Label counts mismatch")

    results = load("results/evaluation_clean_latest.json")
    check(results["n_samples"]==len(rows),"Result denominator mismatch")
    check(results["timing_setup"]["repeats"]==3,"Timing protocol mismatch")
    for name,m in results["models"].items():
        check(m["n_samples"]==len(rows),f"Model denominator: {name}")
        check(abs(m["accuracy"]-m["n_correct"]/len(rows))<=.00005,f"Accuracy: {name}")
        # The saved binary recall, class count, and correct count uniquely
        # determine the confusion counts; verify the other reported metrics.
        tp = round(m["recall"]*pos)
        fn = pos-tp
        tn = m["n_correct"]-tp
        fp = neg-tn
        check(min(tp,fn,tn,fp)>=0,f"Impossible confusion counts: {name}")
        expected={"recall":tp/pos,"precision":tp/(tp+fp),"f1":2*tp/(2*tp+fp+fn)}
        for metric,value in expected.items():
            check(abs(m[metric]-value)<=.00005,f"{metric}: {name}")
        check(m["inference_time_per_sample_ms"]>0,f"Invalid latency: {name}")
    routing=results["models"]["cascade"]["routing"]
    check(routing["threshold"]==.8,"Unexpected routing threshold")
    check(routing["logreg_gate"]+routing["tinybert_gate"]==len(rows),"Routing counts do not sum")
    for name in ["logreg","tinybert"]:
        check(abs(routing[f"{name}_pct"]-100*routing[f"{name}_gate"]/len(rows))<=.05,"Routing percentage mismatch")
    for filename in ["logreg_lime.json","tinybert_lime_clean.json"]:
        explanations=load(f"results/{filename}")
        check(len(explanations)==5,f"Explanation count: {filename}")
        check([r["text"] for r in explanations]==[r["text"] for r in rows[:5]],f"Explanation ordering: {filename}")

    local_links=0
    markdowns=[ROOT/"README.md",ROOT/"NOTICE.md"]+list((ROOT/"docs").glob("*.md"))+list((ROOT/"data").glob("*.md"))+list((ROOT/"models").glob("*.md"))
    for path in markdowns:
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if target.startswith(("http:","https:","mailto:","#")):
                continue
            target=unquote(target.split("#",1)[0].strip("<>"))
            check((path.parent/target).exists(),f"Broken link in {path.name}: {target}")
            local_links+=1

    artifacts=load("models/artifact_manifest.json")
    checked=0
    for relative,metadata in artifacts["files"].items():
        path=ROOT/relative
        if path.exists():
            check(path.stat().st_size==metadata["size_bytes"],f"Artifact size changed: {relative}")
            check(hashlib.sha256(path.read_bytes()).hexdigest()==metadata["sha256"],f"Artifact hash changed: {relative}")
            checked+=1
    print(f"PASS: {len(sources)} Python files; {local_links} local links; sample and metric consistency; LIME alignment; {checked} available model artifacts.")
    if checked<len(artifacts["files"]):
        print("Optional model weights are absent; inference was not checked by this script.")


if __name__=="__main__":
    main()
