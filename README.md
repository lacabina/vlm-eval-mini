# vlm-eval-mini

A lightweight evaluator for vision-language and multimodal QA outputs.

## What it solves
As multimodal models evolve quickly, teams need fast iteration loops for evaluation without heavy infrastructure. This project provides:
- compact dataset format for multimodal QA tasks
- lexical + keyword metrics for answer quality
- modality-aware checks and markdown report export

## Core metrics
- `exact_match`: normalized string exact match
- `token_f1`: overlap-based token F1
- `keyword_recall`: required concept coverage
- `modality_score`: whether response references required modality cues

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Evaluate
```bash
python scripts/evaluate.py \
  --dataset benchmarks/sample_eval.json \
  --predictions examples/mock_predictions.json \
  --output report.md
```

## Output
The evaluator writes a markdown report with per-item and aggregate scores.

## License
MIT
