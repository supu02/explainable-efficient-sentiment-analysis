"""Build GitHub PNGs and report PDFs from the saved clean experiment only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "assets"
COLORS = {"logreg": "#2864A0", "tinybert": "#B77323", "cascade": "#596C3E"}
POSITIVE, NEGATIVE = "#2864A0", "#B77323"


def load(name):
    return json.loads((RESULTS / name).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#AAB2BA")
    ax.tick_params(length=0, pad=8)
    ax.grid(axis="x", color="#E8ECF0", linewidth=0.8)
    ax.set_axisbelow(True)


def latency(results):
    keys = ["logreg", "cascade", "tinybert"]
    labels = ["TF-IDF + Logistic Regression", "Cascade · threshold 0.80", "TinyBERT INT8"]
    values = [results["models"][k]["inference_time_per_sample_ms"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    bars = ax.barh(labels, values, color=[COLORS[k] for k in keys], height=0.53)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.bar_label(bars, labels=[f"{v:.2f} ms" for v in values], padding=8, fontsize=11)
    ax.set_xlabel("Mean CPU inference time per review (ms) · lower is faster", labelpad=12)
    ax.set_title("CPU inference latency", loc="left", weight="bold", fontsize=16, pad=26)
    ax.text(0, 1.06, "39 cleaned reviews · warm-up + 3 repeated per-review passes", transform=ax.transAxes,
            fontsize=10, color="#53606D")
    axes_style(ax)
    save(fig, "latency")


def routing(results):
    routing = results["models"]["cascade"]["routing"]
    fig, ax = plt.subplots(figsize=(8, 3.0), layout="constrained")
    counts = [routing["logreg_gate"], routing["tinybert_gate"]]
    labels = ["Logistic Regression", "TinyBERT fallback"]
    bars = ax.barh(labels, counts, color=[COLORS["logreg"], COLORS["tinybert"]], height=0.52)
    ax.invert_yaxis()
    ax.set_xlim(0, results["n_samples"])
    ax.bar_label(bars, labels=[f"{c}/39 ({100*c/39:.1f}%)" for c in counts], padding=8)
    ax.set_xlabel("Reviews routed to each model (39 total)", labelpad=10)
    ax.set_title(f"Cascade routing · probability threshold {routing['threshold']:.2f}", loc="left", weight="bold", pad=16)
    axes_style(ax)
    save(fig, "cascade_routing")


def diagram(results):
    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1,
                                   edgecolor=color, facecolor="#F7F9FB"))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=11)
    box(.2, 1.1, 1.1, .65, "Review", "#66717C")
    box(2, .9, 3.15, 1.05, "TF-IDF + Logistic Regression\nMaximum probability ≥ 0.80?", COLORS["logreg"])
    box(6.6, 1.85, 2.95, .85, "Yes → return sparse prediction\n22/39 reviews · 56.4%", COLORS["logreg"])
    box(6.6, .25, 2.95, .85, "No → TinyBERT INT8\n17/39 reviews · 43.6%", COLORS["tinybert"])
    for start,end in [((1.4,1.425),(1.9,1.425)),((5.25,1.65),(6.5,2.25)),((5.25,1.15),(6.5,.675))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"->","lw":1.5,"color":"#59636F"})
    save(fig, "cascade_diagram")


def weights(features):
    negative = features["global"]["top_negative"][:8]
    positive = features["global"]["top_positive"][:8]
    rows = sorted(negative + positive, key=lambda r:r["weight"])
    fig, ax = plt.subplots(figsize=(8, 5.9), layout="constrained")
    values = [r["weight"] for r in rows]
    ax.barh([r["word"] for r in rows], values,
            color=[POSITIVE if v >= 0 else NEGATIVE for v in values], height=.68)
    ax.axvline(0, color="#35404A", linewidth=1)
    ax.set_xlabel("Learned Logistic Regression coefficient", labelpad=10)
    ax.set_title("Global sentiment features", loc="left", weight="bold", fontsize=16, pad=26)
    ax.text(0, 1.035, "Negative weights ←                         → Positive weights", transform=ax.transAxes,
            color="#53606D", fontsize=10)
    axes_style(ax)
    save(fig, "feature_weights")


def lime_examples(lr, bert):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout="constrained")
    bound = max(abs(f["weight"]) for data in [lr,bert] for row in data[:2] for f in row["lime_features"][:6]) * 1.18
    for row in range(2):
        assert lr[row]["text"] == bert[row]["text"], "LIME review order mismatch"
        for col, (data, model) in enumerate([(lr,"Logistic Regression"),(bert,"TinyBERT INT8")]):
            ax = axes[row,col]
            features = list(reversed(data[row]["lime_features"][:6]))
            values = [f["weight"] for f in features]
            ax.barh([f["word"] for f in features], values,
                    color=[POSITIVE if v >= 0 else NEGATIVE for v in values], height=.66)
            ax.set_xlim(-bound,bound)
            ax.axvline(0, color="#35404A", linewidth=1)
            ax.set_title(f"Review {row+1} · {model}", loc="left", fontsize=11, weight="bold", pad=10)
            ax.set_xlabel("Local weight toward positive class", fontsize=10)
            axes_style(ax)
    fig.suptitle("Illustrative LIME explanations\nFirst two clean evaluation reviews · one run per model", fontsize=14, weight="bold")
    save(fig, "lime_examples")


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "text.color":"#26323D",
                         "axes.labelcolor":"#394958", "xtick.color":"#53606D", "ytick.color":"#26323D"})
    results = load("evaluation_clean_latest.json")
    latency(results)
    routing(results)
    diagram(results)
    weights(load("logreg_features.json"))
    lime_examples(load("logreg_lime.json"),load("tinybert_lime_clean.json"))
    print(f"Generated 5 figures in PNG and PDF at {OUT}")


if __name__ == "__main__":
    main()
