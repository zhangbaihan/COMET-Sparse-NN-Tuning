import sys
sys.dont_write_bytecode = True

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import get_max_and_min_arrays


def parse_args():
    parser = argparse.ArgumentParser(description="Compare COMET vs COMET_affine validation accuracy over epochs")
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--neurons", type=int, default=500)
    parser.add_argument("--topk", type=float, default=0.9)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models", nargs="+", type=str, default=["COMET_model", "COMET_affine"])
    parser.add_argument("--output", type=str, default=None, help="Optional output path for the PDF")
    return parser.parse_args()


def main():
    args = parse_args()

    results_path = os.path.join(
        args.results_dir,
        args.activation,
        args.optimizer,
        f"lr_{args.lr}",
        f"{args.dataset}_topk_{args.topk}_neurons_{args.neurons}",
    )
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at: {results_path}")

    with open(results_path, "rb") as fp:
        performances = pickle.load(fp)

    # Prepare plot
    plt.figure(figsize=(10, 6))
    colors = {
        "COMET_model": "#1f77b4",       # blue
        "COMET_affine": "#ff7f0e",      # orange
    }
    handles = []
    labels = []

    for model_name in args.models:
        key = f"{model_name}_val_acc"
        if key not in performances:
            print(f"Warning: {key} not found in results. Skipping.")
            continue
        val_acc_across_seeds = performances[key]  # List[List[float]]: seeds x epochs
        max_seed, min_seed, _ = get_max_and_min_arrays(val_acc_across_seeds)
        mean_curve = (np.array(max_seed) + np.array(min_seed)) / 2.0
        epochs = list(range(1, len(mean_curve) + 1))

        color = colors.get(model_name, None)
        fill = plt.fill_between(epochs, np.array(max_seed), np.array(min_seed),
                                alpha=0.3, linewidth=2.0, color=color)
        line, = plt.plot(epochs, mean_curve, linewidth=2, color=color)
        handles.append(line)
        labels.append(model_name)

    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Validation Accuracy", fontsize=14)
    title = f"{args.dataset} | neurons={args.neurons}, p_k={args.topk}, opt={args.optimizer}, lr={args.lr}, act={args.activation}"
    plt.title(title, fontsize=14)
    plt.legend(handles=handles, labels=labels, loc="lower right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()

    if args.output is None:
        out_name = f"compare_comet_vs_affine_{args.dataset}_neurons_{args.neurons}_topk_{args.topk}_opt_{args.optimizer}_lr_{args.lr}_{args.activation}.pdf"
        args.output = out_name
    plt.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()


