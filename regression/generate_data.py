import os
import sys

sys.path.append(".")

import numpy as np
import torch

from itertools import product
from tqdm import tqdm

from data import get_batch_with_prob, DATASETS, TASKS, _DATASET_CONFIGS


DRYRUN = False
PLOT = True
BATCH_SIZE = 4
DATASET_SIZE = {"training": int(2**14), "interpolation": 128}
SEED = 0

# Output directory (next to this script): regression/data
BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)


if PLOT:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6 + 2, 3, figsize=(8, 20))
    axes = np.array(axes).flatten()
    j = 0


# Torch RNG
g = torch.Generator()
g.manual_seed(SEED)


for dataset, task in product(DATASETS, TASKS):
    for input_dim in range(1, _DATASET_CONFIGS[dataset].max_input_dim + 1):
        print(dataset, task, input_dim)

        batches = []
        for _ in tqdm(range(DATASET_SIZE[task] // BATCH_SIZE)):
            batch = get_batch_with_prob(
                g,
                batch_size=BATCH_SIZE,
                name=dataset,
                task=task,
                input_dim=input_dim,
            )
            batches.append(batch)

        # Concatenate along batch dimension
        x_context = torch.cat([b.x_context for b in batches], dim=0)
        y_context = torch.cat([b.y_context for b in batches], dim=0)
        x_target = torch.cat([b.x_target for b in batches], dim=0)
        y_target = torch.cat([b.y_target for b in batches], dim=0)
        mask_target = torch.cat([b.mask_target for b in batches], dim=0)
        mask_context = torch.cat([b.mask_context for b in batches], dim=0)

        print(f"{dataset} {input_dim} {task}")
        print(
            x_context.shape,
            y_context.shape,
            x_target.shape,
            y_target.shape,
            mask_target.shape,
            mask_context.shape,
        )

        if not DRYRUN:
            out_path = os.path.join(OUT_DIR, f"{dataset}_{input_dim}_{task}.npz")
            np.savez(
                out_path,
                x_context=x_context.numpy(),
                y_context=y_context.numpy(),
                x_target=x_target.numpy(),
                y_target=y_target.numpy(),
                mask_target=mask_target.numpy(),
                mask_context=mask_context.numpy(),
            )

        if PLOT:
            # number of unmasked points per example
            num_context = mask_context.shape[1] - mask_context.bool().sum(
                dim=1, keepdim=True
            )
            num_context = num_context.view(-1).cpu().numpy()

            num_target = mask_target.shape[1] - mask_target.bool().sum(
                dim=1, keepdim=True
            )
            num_target = num_target.view(-1).cpu().numpy()

            axes[j].hist(num_context, bins=20, label="context")
            axes[j].hist(num_target, bins=20, label="target")
            axes[j].set_title(f"{dataset} {input_dim} {task}", fontsize=8)
            if j == 0:
                axes[j].legend()
            j += 1

if PLOT:
    plt.savefig("num_data.png")
