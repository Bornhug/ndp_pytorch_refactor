"""CLI for fixed-task finetuning across all filtered TabICL_regression03 tasks."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuning.common import (
    add_shared_cli_args,
    aggregate_metric_dicts,
    parse_dataset_names,
    resolve_filtered_dataset_names,
    run_dataset_finetune_suite,
    write_json,
)


def _aggregate_overall_metrics(
    dataset_summaries: list[dict],
    *,
    metric_field: str,
) -> dict[str, float | list[float]]:
    fold_values: dict[int, dict[str, list[float]]] = {}
    for summary in dataset_summaries:
        for result in summary.get("completed", []):
            fold_index = int(result.get("fold_index", 1))
            metric_dict = result.get(metric_field, {})
            bucket = fold_values.setdefault(fold_index, {"R2": [], "RMSE": [], "MAE": []})
            for key in ("R2", "RMSE", "MAE"):
                if key in metric_dict:
                    bucket[key].append(float(metric_dict[key]))

    per_fold_metric_dicts: list[dict[str, float]] = []
    for fold_index in sorted(fold_values):
        metric_dict: dict[str, float] = {}
        for key in ("R2", "RMSE", "MAE"):
            values = fold_values[fold_index][key]
            if values:
                metric_dict[key] = float(sum(values) / len(values))
        if metric_dict:
            per_fold_metric_dicts.append(metric_dict)
    return aggregate_metric_dicts(per_fold_metric_dicts)


def _build_summary_payload(
    *,
    args,
    output_root: Path,
    requested_datasets: list[str] | None,
    resolved_datasets: list[str],
    completed: list[dict],
    dataset_summaries: list[dict],
    failures: list[dict],
    summary_path: Path | None = None,
) -> dict:
    payload = {
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "output_root": str(output_root),
        "requested_datasets": requested_datasets,
        "resolved_datasets": resolved_datasets,
        "n_splits": int(args.n_splits),
        "completed": completed,
        "dataset_summaries": dataset_summaries,
        "overall_base_metrics": _aggregate_overall_metrics(
            dataset_summaries,
            metric_field="base_metrics",
        ),
        "overall_finetuned_metrics": _aggregate_overall_metrics(
            dataset_summaries,
            metric_field="finetuned_metrics",
        ),
        "overall_metric_deltas": _aggregate_overall_metrics(
            dataset_summaries,
            metric_field="metric_deltas",
        ),
        "failures": failures,
    }
    if summary_path is not None:
        payload["summary_json"] = str(summary_path)
    return payload


def main() -> dict:
    parser = argparse.ArgumentParser(
        description="Finetune TabICL_regression03 on every filtered evaluation task."
    )
    add_shared_cli_args(parser, include_dataset=False)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated subset of evaluation datasets.",
    )
    args = parser.parse_args()

    requested_datasets = parse_dataset_names(args.datasets)
    dataset_names = resolve_filtered_dataset_names(
        max_features_eval=int(args.max_features_eval),
        new_instances_eval=int(args.new_instances_eval),
        random_state=int(args.random_state),
        use_cache=not bool(args.no_cache),
        verbose=bool(args.verbose),
        dataset_names=requested_datasets,
    )

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"

    results: list[dict] = []
    dataset_summaries: list[dict] = []
    failures: list[dict] = []
    total = len(dataset_names)

    for idx, dataset_name in enumerate(dataset_names, start=1):
        task_output_dir = output_root / dataset_name
        print(
            f"[{idx}/{total}] Finetuning {dataset_name} -> {task_output_dir}",
            flush=True,
        )
        try:
            dataset_summary = run_dataset_finetune_suite(
                args,
                dataset_name=dataset_name,
                output_dir=task_output_dir,
            )
            dataset_summaries.append(dataset_summary)
            results.extend(dataset_summary.get("completed", []))
            failures.extend(dataset_summary.get("failures", []))
        except Exception as exc:
            failures.append(
                {
                    "dataset_name": dataset_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Task failed for {dataset_name}: {exc}", file=sys.stderr, flush=True)

        summary_payload = _build_summary_payload(
            args=args,
            output_root=output_root,
            requested_datasets=requested_datasets,
            resolved_datasets=dataset_names,
            completed=results,
            dataset_summaries=dataset_summaries,
            failures=failures,
        )
        write_json(summary_path, summary_payload)

    final_payload = _build_summary_payload(
        args=args,
        output_root=output_root,
        requested_datasets=requested_datasets,
        resolved_datasets=dataset_names,
        completed=results,
        dataset_summaries=dataset_summaries,
        failures=failures,
        summary_path=summary_path,
    )
    write_json(summary_path, final_payload)
    print(json.dumps(final_payload, indent=2, sort_keys=True))

    if failures:
        raise SystemExit(1)
    return final_payload


if __name__ == "__main__":
    main()
