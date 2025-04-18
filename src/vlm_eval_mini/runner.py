import json
from pathlib import Path

from .metrics import exact_match, keyword_recall, modality_score, token_f1


def evaluate_predictions(dataset_path: str | Path, predictions_path: str | Path) -> dict:
    dataset = json.loads(Path(dataset_path).read_text())
    predictions = json.loads(Path(predictions_path).read_text())
    pred_map = {item["id"]: item["prediction"] for item in predictions}

    rows = []
    for sample in dataset:
        sample_id = sample["id"]
        pred = pred_map.get(sample_id, "")
        ref = sample["reference"]

        em = exact_match(pred, ref)
        f1 = token_f1(pred, ref)
        kw = keyword_recall(pred, sample.get("required_keywords", []))
        mod = modality_score(pred, sample.get("required_modalities", []))
        total = 0.2 * em + 0.4 * f1 + 0.25 * kw + 0.15 * mod

        rows.append(
            {
                "id": sample_id,
                "question": sample["question"],
                "prediction": pred,
                "reference": ref,
                "scores": {
                    "exact_match": round(em, 4),
                    "token_f1": round(f1, 4),
                    "keyword_recall": round(kw, 4),
                    "modality_score": round(mod, 4),
                    "overall": round(total, 4),
                },
            }
        )

    if rows:
        avg = {
            metric: round(sum(item["scores"][metric] for item in rows) / len(rows), 4)
            for metric in ["exact_match", "token_f1", "keyword_recall", "modality_score", "overall"]
        }
    else:
        avg = {k: 0.0 for k in ["exact_match", "token_f1", "keyword_recall", "modality_score", "overall"]}

    return {"summary": avg, "details": rows}


def write_markdown_report(results: dict, output_path: str | Path) -> None:
    lines = ["# VLM Eval Mini Report", "", "## Summary", ""]
    for k, v in results["summary"].items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## Details", ""])
    for row in results["details"]:
        lines.append(f"### {row['id']}: {row['question']}")
        lines.append(f"- overall: {row['scores']['overall']}")
        lines.append(f"- prediction: {row['prediction']}")
        lines.append(f"- reference: {row['reference']}")
        lines.append("")

    Path(output_path).write_text("\n".join(lines))
