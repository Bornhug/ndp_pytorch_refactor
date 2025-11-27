from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_npz(
    npz_path: str,
    out_path: Optional[str] = None,
    max_batches: int = 16,
) -> None:
    data = np.load(npz_path)

    x_target = data["x_target"]        # [B, N_T, D]
    y_target = data["y_target"]        # [B, N_T, 1]
    x_context = data["x_context"]      # [B, N_C, D]
    y_context = data["y_context"]      # [B, N_C, 1]
    mask_target = data["mask_target"]  # [B, N_T]
    mask_context = data["mask_context"]  # [B, N_C]

    if x_target.ndim != 3 or y_target.ndim != 3:
        raise ValueError("Expected x_target/y_target with shape [B, N, D] / [B, N, 1].")
    if x_target.shape[-1] != 1:
        raise ValueError("plot_npz currently supports input_dim=1 (last dim = 1).")

    B = x_target.shape[0]
    num_batches = min(B, max_batches)

    cols = min(4, num_batches)
    rows = math.ceil(num_batches / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()

    for b in range(num_batches):
        ax = axes[b]

        xt = x_target[b, :, 0]
        yt = y_target[b, :, 0]
        mt = mask_target[b]

        xc = x_context[b, :, 0]
        yc = y_context[b, :, 0]
        mc = mask_context[b]

        keep_t = mt == 0  # unmasked targets
        keep_c = mc == 0  # unmasked context

        # Determine conditional vs unconditional from context mask
        has_context = np.any(keep_c)

        # Plot targets (blue), skipping masked positions
        ax.scatter(xt[keep_t], yt[keep_t], s=10, color="C0", label="target")

        # Plot context only for conditional batches
        if has_context:
            ax.scatter(xc[keep_c], yc[keep_c], s=20, color="k", marker="o", label="context")

        ax.set_title(f"batch {b} ({'cond' if has_context else 'uncond'})", fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Hide any unused axes
    for b in range(num_batches, len(axes)):
        fig.delaxes(axes[b])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot samples from an NDP regression .npz dataset.")
    parser.add_argument("npz_path", help="Path to .npz file under regression/data.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (PNG). If omitted, shows an interactive window.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=16,
        help="Maximum number of batches to plot.",
    )

    args = parser.parse_args(argv)
    plot_npz(args.npz_path, out_path=args.out, max_batches=args.max_batches)


if __name__ == "__main__":
    main()

