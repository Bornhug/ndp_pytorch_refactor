"""CLI for fixed-task finetuning on one TabICL_regression03 evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuning.common import add_shared_cli_args, run_single_task_finetune


def main() -> dict:
    parser = argparse.ArgumentParser(
        description="Finetune TabICL_regression03 on one fixed evaluation task."
    )
    add_shared_cli_args(parser, include_dataset=True)
    args = parser.parse_args()

    summary = run_single_task_finetune(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("failures"):
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
