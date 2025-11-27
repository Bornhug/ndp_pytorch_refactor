import argparse
import os
import sys

import numpy as np
import pandas as pd


def npz_to_excel(npz_path: str, excel_path: str | None = None) -> str:
    """
    Convert a regression .npz file to an Excel workbook with one sheet per array.
    Defaults to writing under the visualization directory next to this script.
    """
    if excel_path is None:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        excel_path = os.path.join(os.path.dirname(__file__), f"{base_name}.xlsx")

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    data = np.load(npz_path)

    with pd.ExcelWriter(excel_path) as writer:
        for name in data.files:
            arr = data[name]
            arr2d = arr.reshape(arr.shape[0], -1)
            df = pd.DataFrame(arr2d)
            df.to_excel(writer, sheet_name=name[:31], index=False)

    return excel_path


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Convert an NDP regression .npz to Excel.")
    parser.add_argument("npz_path", help="Path to .npz file (e.g., regression/data/matern_1_training.npz).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output Excel path. Defaults to regression/visualization/<npz_basename>.xlsx",
    )
    args = parser.parse_args(argv[1:])

    out = npz_to_excel(args.npz_path, excel_path=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv)
