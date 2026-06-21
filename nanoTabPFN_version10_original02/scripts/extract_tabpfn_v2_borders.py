from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bar_distribution import DEFAULT_BORDERS_PATH, DEFAULT_NUM_BARS


def _build_regressor(version: str, device: str):
    from tabpfn import TabPFNRegressor

    if hasattr(TabPFNRegressor, "create_default_for_version"):
        try:
            return TabPFNRegressor.create_default_for_version(
                version,
                device=device,
                n_estimators=1,
                show_progress_bar=False,
            )
        except TypeError:
            return TabPFNRegressor.create_default_for_version(version)
    return TabPFNRegressor(device=device, n_estimators=1, show_progress_bar=False)


def extract_borders(*, version: str, device: str) -> torch.Tensor:
    regressor = _build_regressor(version, device)
    if not hasattr(regressor, "_initialize_model_variables"):
        raise RuntimeError("Installed TabPFNRegressor has no _initialize_model_variables hook.")
    regressor._initialize_model_variables()
    bardist = getattr(regressor, "znorm_space_bardist_", None)
    if bardist is None:
        raise RuntimeError("TabPFNRegressor did not expose znorm_space_bardist_.")
    borders = getattr(bardist, "borders", None)
    if borders is None:
        raise RuntimeError("TabPFNRegressor znorm_space_bardist_ has no borders tensor.")
    return torch.as_tensor(borders, dtype=torch.float32).detach().cpu().flatten().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract official TabPFN v2 regressor normalized bar borders."
    )
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-bars", type=int, default=DEFAULT_NUM_BARS)
    parser.add_argument("--output", type=str, default=str(DEFAULT_BORDERS_PATH))
    args = parser.parse_args()

    borders = extract_borders(version=args.version, device=args.device)
    expected_count = int(args.num_bars) + 1
    if borders.numel() != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} borders for {args.num_bars} bars, got {borders.numel()}."
        )
    if not torch.isfinite(borders).all():
        raise RuntimeError("Extracted borders contain non-finite values.")
    if not torch.all(borders[1:] > borders[:-1]):
        raise RuntimeError("Extracted borders are not strictly increasing.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "borders": borders,
            "source": "official_tabpfn_regressor_znorm_space_bardist",
            "tabpfn_version": args.version,
            "num_bars": int(args.num_bars),
        },
        output,
    )
    print(f"Saved {borders.numel()} borders to {output}")
    print(f"range=[{float(borders[0]):.6g}, {float(borders[-1]):.6g}]")


if __name__ == "__main__":
    main()
