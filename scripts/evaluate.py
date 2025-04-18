import argparse
import json

from vlm_eval_mini.runner import evaluate_predictions, write_markdown_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="report.md")
    args = parser.parse_args()

    results = evaluate_predictions(args.dataset, args.predictions)
    write_markdown_report(results, args.output)

    print(json.dumps(results["summary"], indent=2))
    print(f"Report written to {args.output}")
