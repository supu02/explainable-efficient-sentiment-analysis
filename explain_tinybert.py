"""Generate local LIME explanations for the retained quantized TinyBERT model.

Completed workflow:
    python explain_tinybert.py --examples data/eval_clean.csv --method lime

The inherited IG/both options are unfinished with the available artifact set:
no compatible fine-tuned float checkpoint is included. Use LIME for this project.
The saved outputs are qualitative examples, not a stability or faithfulness study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _run_ig(texts: list, top_n: int, n_steps: int) -> list:
    """
    Integrated Gradients explanations for TinyBERT.

    Requires a float (non-quantized) checkpoint. The active checkpoint
    (models/tinybert_quantized.pt) was saved *after* INT8 quantization and
    contains quantized-format keys (_packed_params, scale, zero_point) that
    cannot be loaded into a float BertForSequenceClassification.

    This function raises RuntimeError immediately with an actionable message
    rather than crashing with an opaque key-mismatch error from load_state_dict.

    To enable IG in a future training run, save the float model *before*
    calling _quantize_model() in TinyBertModel.train(), then point this
    function at that float checkpoint.
    """
    from src.config import TINYBERT_MODEL_PATH

    # ── Pre-flight: detect quantized state dict ──────────────────────────────
    checkpoint = torch.load(TINYBERT_MODEL_PATH, map_location="cpu", weights_only=False)
    _QUANT_MARKERS = ("_packed_params", "zero_point")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        sd_keys = list(checkpoint["model_state_dict"].keys())
        is_quantized = any(
            any(m in k for m in _QUANT_MARKERS) for k in sd_keys
        )
        if is_quantized:
            raise RuntimeError(
                f"Integrated Gradients is not available: {TINYBERT_MODEL_PATH} "
                "was saved after INT8 quantization (state-dict contains "
                "'_packed_params' / 'zero_point' keys). "
                "No float checkpoint exists in the repo.\n"
                "To fix: in TinyBertModel.train_from_csv(), save the model once "
                "before the _quantize_model() call, then pass that path via a "
                "future --ig-model-path flag. "
                "For now, run with --method lime."
            )

    try:
        from captum.attr import LayerIntegratedGradients  # type: ignore
    except ImportError:
        raise ImportError(
            "captum not found. Install it with: pip install captum"
        )

    from src.tinybert_model import TinyBertModel, TinyBertConfig, IMDbDataset
    from src.config import TINYBERT_TOKENIZER_PATH
    from transformers import AutoTokenizer

    # ── Load float (non-quantized) model copy ───────────────────────────────
    print("[explain_tinybert] Loading float model for Integrated Gradients...")
    fw = TinyBertModel(load=False)
    fw.config = TinyBertConfig(quantize=False)
    fw.training_mode = False
    fw.device = torch.device("cpu")
    fw.tokenizer = AutoTokenizer.from_pretrained(TINYBERT_TOKENIZER_PATH)
    fw.model = fw._build_model()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        fw.model.load_state_dict(checkpoint["model_state_dict"])
    else:
        fw.model = checkpoint  # legacy full-model format
    fw.model.eval()

    bert_model = fw.model
    tokenizer = fw.tokenizer

    # ── Captum forward function (uses inputs_embeds to allow grad flow) ─────
    def _forward_embeds(input_embeds, attention_mask):
        return bert_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).logits

    lig = LayerIntegratedGradients(
        _forward_embeds,
        bert_model.bert.embeddings.word_embeddings,
    )

    results = []
    for i, text in enumerate(texts):
        text_clean = IMDbDataset._clean_text(text)
        encoding = fw._head_tail_tokenize_inference(text_clean)

        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([encoding["attention_mask"]], dtype=torch.long)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

        # Predicted class used as attribution target
        with torch.no_grad():
            logits = bert_model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred = int(torch.argmax(logits, dim=-1).item())

        # Compute attributions
        attributions, delta = lig.attribute(
            input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=pred,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        # Sum over embedding dim → scalar per token
        token_attrs = attributions.squeeze(0).sum(dim=-1).detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        # Filter special tokens, sort by absolute attribution
        SPECIALS = {"[CLS]", "[SEP]", "[PAD]"}
        pairs = [
            (tok, float(attr))
            for tok, attr in zip(tokens, token_attrs.tolist())
            if tok not in SPECIALS
        ]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        label_str = "positive" if pred == 1 else "negative"
        ig_features = [
            {"token": tok, "attribution": round(attr, 4)}
            for tok, attr in pairs[:top_n]
        ]
        results.append({
            "text": text,
            "prediction": pred,
            "prediction_label": label_str,
            "convergence_delta": round(float(delta.mean().item()), 6),
            "ig_features": ig_features,
        })

        print(f"  [{i}] {label_str}: {text[:70]}{'...' if len(text) > 70 else ''}")
        for feat in ig_features[:5]:
            print(f"        {feat['attribution']:+.4f}  {feat['token']}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TinyBERT LIME explanations (IG unavailable with supplied weights)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--examples", required=True,
        help="CSV file with a 'text' column to explain",
    )
    parser.add_argument(
        "--method", default="lime", choices=["lime", "ig", "both"],
        help="Explanation method: lime | ig | both",
    )
    parser.add_argument(
        "--n-examples", type=int, default=5,
        help="Number of rows from the CSV to explain",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of features/tokens per example",
    )
    parser.add_argument(
        "--num-samples", type=int, default=300,
        help="LIME perturbation samples (lower = faster, less stable)",
    )
    parser.add_argument(
        "--ig-steps", type=int, default=50,
        help="IG interpolation steps (higher = more accurate, slower)",
    )
    parser.add_argument(
        "--out", default="results/tinybert_lime_clean.json",
        help="Output path for the LIME JSON file",
    )
    parser.add_argument(
        "--ig-out", default="results/tinybert_ig.json",
        help="Output path for the Integrated Gradients JSON file",
    )
    args = parser.parse_args()

    # ── Validate input ────────────────────────────────────────────────────────
    csv_path = Path(args.examples)
    if not csv_path.exists():
        raise FileNotFoundError(f"Examples file not found: {csv_path}")

    # ── Load examples ─────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError(f"CSV must contain a 'text' column; found: {list(df.columns)}")
    texts = df["text"].astype(str).tolist()[: args.n_examples]

    run_lime = args.method in ("lime", "both")
    run_ig   = args.method in ("ig",   "both")

    # ── LIME ─────────────────────────────────────────────────────────────────
    if run_lime:
        from src.tinybert_model import TinyBertModel, IMDbDataset

        print("[explain_tinybert] Loading TinyBertModel (quantized) for LIME...")
        model = TinyBertModel()
        model.model.eval()

        # LIME expects: List[str] -> np.ndarray of shape (n, 2)
        def _predict_proba(batch_texts: list) -> np.ndarray:
            probs = []
            for t in batch_texts:
                t_clean = IMDbDataset._clean_text(t)
                enc = model._head_tail_tokenize_inference(t_clean)
                input_ids = torch.tensor(
                    [enc["input_ids"]], dtype=torch.long
                ).to(model.device)
                attention_mask = torch.tensor(
                    [enc["attention_mask"]], dtype=torch.long
                ).to(model.device)
                with torch.no_grad():
                    logits = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits
                    prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                probs.append(prob)
            return np.array(probs)

        try:
            from lime.lime_text import LimeTextExplainer  # type: ignore
        except ImportError:
            raise ImportError("lime not found. Install: pip install lime")

        lime_explainer = LimeTextExplainer(class_names=["negative", "positive"])
        print(f"[explain_tinybert] Running LIME on {len(texts)} examples "
              f"(top {args.top_n} features, {args.num_samples} samples each)...")

        lime_results = []
        for i, text in enumerate(texts):
            exp = lime_explainer.explain_instance(
                text,
                _predict_proba,
                num_features=args.top_n,
                num_samples=args.num_samples,
            )
            pred = model.predict(text)
            label_str = "positive" if pred == 1 else "negative"
            lime_features = [
                {"word": word, "weight": round(float(weight), 4)}
                for word, weight in exp.as_list()
            ]
            lime_results.append({
                "text": text,
                "prediction": pred,
                "prediction_label": label_str,
                "lime_features": lime_features,
            })
            print(f"  [{i}] {label_str}: {text[:70]}{'...' if len(text) > 70 else ''}")
            for feat in lime_features[:5]:
                print(f"        {feat['weight']:+.4f}  {feat['word']}")

        out_path = Path(args.out)
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(lime_results, indent=2))
        print(f"\n[explain_tinybert] LIME saved → {out_path}")

    # ── Integrated Gradients ──────────────────────────────────────────────────
    if run_ig:
        print(f"\n[explain_tinybert] Running Integrated Gradients on {len(texts)} examples "
              f"(top {args.top_n} tokens, {args.ig_steps} steps each)...")
        ig_results = _run_ig(texts, args.top_n, args.ig_steps)

        ig_out_path = Path(args.ig_out)
        ig_out_path.parent.mkdir(exist_ok=True)
        ig_out_path.write_text(json.dumps(ig_results, indent=2))
        print(f"\n[explain_tinybert] IG saved → {ig_out_path}")


if __name__ == "__main__":
    main()
