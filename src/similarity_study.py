"""
Similarity Study: Track dictionary similarity during SAE training.

Trains multiple SI-SAE models with controlled seeds and tracks:
1. Similarity between each model and its initialization (with noise)
2. Similarity between each model and the clean k-means centers (without noise)
3. Pairwise similarity between models with same init+noise (same seed)
4. Pairwise similarity between models with different noise (different seed)

Results are saved as JSON and figures are generated automatically.

Usage:
    python similarity_study.py <shard_directory> [options]
    python similarity_study.py --plot-only path/to/similarity_study_results.json [--output fig.pdf]
"""

import os
import sys
import json
import time
import argparse
import glob as globmod
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import create_dataloader, create_val_dataloader, DeviceDataLoader
from utils import cosine_kmeans
from overcomplete.sae import TopKSAE
from overcomplete.sae.train import extract_input
from overcomplete.metrics import l2, l0_eps
from overcomplete.sae.trackers import DeadCodeTracker
from train import EarlyStopping, validate
from common import GracefulKiller, criterion, create_optimizer_scheduler


# ── Similarity metrics ──────────────────────────────────────────────────────

def mean_max_cosine_similarity(D1, D2):
    """
    Fast similarity: for each atom in D1, find the max cosine sim to any atom in D2.
    Returns the average of these maxima. Symmetric version averages both directions.
    """
    D1_norm = D1 / (D1.norm(dim=1, keepdim=True) + 1e-6)
    D2_norm = D2 / (D2.norm(dim=1, keepdim=True) + 1e-6)
    sim = torch.matmul(D1_norm, D2_norm.T)  # (n1, n2)
    max_1to2 = sim.max(dim=1).values.mean().item()
    max_2to1 = sim.max(dim=0).values.mean().item()
    return (max_1to2 + max_2to1) / 2


def cosine_similarity_matrix(D1, D2):
    """Full cosine similarity matrix between two dictionaries."""
    D1_norm = D1 / (D1.norm(dim=1, keepdim=True) + 1e-6)
    D2_norm = D2 / (D2.norm(dim=1, keepdim=True) + 1e-6)
    return torch.matmul(D1_norm, D2_norm.T)


# ── SI-SAE creation with controlled seed ────────────────────────────────────

def create_si_sae(d_brain, d_model, k, device, centers, per_init, init_seed):
    """
    Create an SI-SAE with a specific random seed controlling the noise.

    Parameters
    ----------
    centers : torch.Tensor
        Clean k-means centers (d_model, d_brain).
    per_init : float
        Noise level for SI initialization.
    init_seed : int
        Random seed for the noise generation.

    Returns
    -------
    sae : TopKSAE
        The initialized model.
    init_dict : torch.Tensor
        The initial dictionary weights (for similarity tracking).
    """
    # Create base model (seed doesn't matter much here, we overwrite weights)
    sae = TopKSAE(input_shape=d_brain, nb_concepts=d_model, top_k=k, device=device)

    # Generate noise with controlled seed
    rng = torch.Generator()
    rng.manual_seed(init_seed)
    noise = torch.randn(centers.shape, generator=rng) * centers.std() * per_init
    weights = (1 - per_init) * centers + noise

    # Set dictionary and encoder
    norm_weights = torch.nn.functional.normalize(weights.to(device), p=2, dim=-1)
    sae.dictionary._weights.data = norm_weights
    enc_weights = norm_weights.clone() * (1.0 / (k ** 0.5))
    sae.encoder.final_block[0].weight.data = enc_weights

    init_dict = norm_weights.detach().clone().cpu()
    return sae, init_dict


# ── Training with similarity tracking ───────────────────────────────────────

def train_with_similarity_tracking(
    model, dataloader, optimizer, scheduler,
    nb_epochs, device, model_name,
    init_dict, clean_centers,
    other_models=None,
    clip_grad=1.0,
    use_mixed_precision=False,
    freeze_dict_epochs=2,
    val_loader=None,
):
    """
    Train an SAE and track dictionary similarity at each epoch.

    Parameters
    ----------
    init_dict : torch.Tensor
        Initial dictionary weights (with noise) on CPU.
    clean_centers : torch.Tensor
        Clean k-means centers (without noise) on CPU.
    other_models : dict, optional
        {name: model} dict of other models to compute pairwise similarity with.

    Returns
    -------
    logs : dict
        Training logs including similarity trajectories.
    """
    logs = defaultdict(list)
    global_step = 0
    frozen = freeze_dict_epochs > 0

    if frozen:
        for param in model.dictionary.parameters():
            param.requires_grad = False

    # Move references to device for similarity computation
    init_dict_dev = init_dict.to(device)
    clean_centers_dev = clean_centers.to(device)

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    for epoch in range(nb_epochs):
        if frozen and epoch >= freeze_dict_epochs:
            for param in model.dictionary.parameters():
                param.requires_grad = True
            print(f"  [{model_name}] Unfreezing dictionary at epoch {epoch+1}")
            frozen = False

        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        mon_count = 0
        dead_tracker = None

        for batch in dataloader:
            global_step += 1
            batch_count += 1
            x = extract_input(batch)
            optimizer.zero_grad(set_to_none=True)

            if use_mixed_precision:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    z_pre, z, x_hat = model(x)
                loss = criterion(x.float(), x_hat.float(), z_pre.float(),
                                 z.float(), model.get_dictionary().float())
                if dead_tracker is None:
                    dead_tracker = DeadCodeTracker(z.shape[1], device)
                dead_tracker.update(z.float())
                loss.backward()
            else:
                x = x.float()
                z_pre, z, x_hat = model(x)
                loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
                if dead_tracker is None:
                    dead_tracker = DeadCodeTracker(z.shape[1], device)
                dead_tracker.update(z)
                loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if batch_count % 50 == 0:
                mon_count += 1
                epoch_loss += loss.item()

        epoch_duration = time.time() - start_time

        # ── Compute similarities at end of epoch ──
        model.eval()
        with torch.no_grad():
            D = model.get_dictionary().detach()

            sim_to_init = mean_max_cosine_similarity(D, init_dict_dev)
            sim_to_clean = mean_max_cosine_similarity(D, clean_centers_dev)

            logs['sim_to_init'].append(sim_to_init)
            logs['sim_to_clean_centers'].append(sim_to_clean)

            # Pairwise similarities with other models
            if other_models:
                for other_name, other_model in other_models.items():
                    other_model.eval()
                    D_other = other_model.get_dictionary().detach()
                    pair_sim = mean_max_cosine_similarity(D, D_other)
                    logs[f'sim_to_{other_name}'].append(pair_sim)

        avg_loss = epoch_loss / mon_count if mon_count > 0 else float('nan')
        dead_ratio = dead_tracker.get_dead_ratio() if dead_tracker else 0.0
        logs['avg_loss'].append(avg_loss)
        logs['dead_features'].append(dead_ratio)
        logs['time_epoch'].append(epoch_duration)

        # Validation
        val_msg = ""
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
            logs['val_loss'].append(val_metrics['val_loss'])
            val_msg = f" | Val loss: {val_metrics['val_loss']:.4f}"

        pair_str = ""
        if other_models:
            pair_strs = [f"{k}: {logs[k][-1]:.4f}" for k in logs if k.startswith('sim_to_') and k not in ('sim_to_init', 'sim_to_clean_centers')]
            if pair_strs:
                pair_str = " | " + ", ".join(pair_strs)

        print(f"  [{model_name}] Epoch {epoch+1:3d}/{nb_epochs} "
              f"Loss: {avg_loss:.4f} Dead: {dead_ratio*100:.1f}% "
              f"SimInit: {sim_to_init:.4f} SimClean: {sim_to_clean:.4f}"
              f"{pair_str}{val_msg} ({epoch_duration:.1f}s)")

    return dict(logs)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_similarity_results(results, output_path):
    """Generate similarity trajectory figures from study results."""
    config = results["config"]
    logs_a = results["model_A"]
    logs_b = results["model_B"]
    logs_c = results["model_C"]

    epochs_a = list(range(1, len(logs_a.get("sim_to_init", [])) + 1))
    epochs_b = list(range(1, len(logs_b.get("sim_to_init", [])) + 1))
    epochs_c = list(range(1, len(logs_c.get("sim_to_init", [])) + 1))

    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Dictionary Similarity During Training\n"
        f"(d_model={config['d_model']}, k={config['k']}, per_init={config['per_init']})",
        fontsize=14, fontweight='bold'
    )

    # Panel 1: Model vs Init (with noise)
    ax = axes[0, 0]
    ax.plot(epochs_a, logs_a["sim_to_init"], color=colors['A'], linewidth=2, label="A (seed=42)")
    ax.plot(epochs_b, logs_b["sim_to_init"], color=colors['B'], linewidth=2, label="B (seed=42)")
    ax.plot(epochs_c, logs_c["sim_to_init"], color=colors['C'], linewidth=2, label="C (seed=43)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Model vs Its Own Init (with noise)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    # Panel 2: Model vs Clean Centers
    ax = axes[0, 1]
    ax.plot(epochs_a, logs_a["sim_to_clean_centers"], color=colors['A'], linewidth=2, label="A (seed=42)")
    ax.plot(epochs_b, logs_b["sim_to_clean_centers"], color=colors['B'], linewidth=2, label="B (seed=42)")
    ax.plot(epochs_c, logs_c["sim_to_clean_centers"], color=colors['C'], linewidth=2, label="C (seed=43)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Model vs Clean Centers (no noise)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    # Panel 3: Same init + same noise (A ↔ B)
    ax = axes[1, 0]
    if "sim_to_A" in logs_b:
        ax.plot(epochs_b, logs_b["sim_to_A"], color='#d62728',
                linewidth=2, label="B ↔ A (same init, same noise)")
    ax.set_xlabel("Epoch (of model B)")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Pairwise: Same Init + Same Noise (A ↔ B)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    # Panel 4: Same init + different noise (C ↔ A, C ↔ B)
    ax = axes[1, 1]
    if "sim_to_A" in logs_c:
        ax.plot(epochs_c, logs_c["sim_to_A"], color='#9467bd',
                linewidth=2, label="C ↔ A (different noise)")
    if "sim_to_B" in logs_c:
        ax.plot(epochs_c, logs_c["sim_to_B"], color='#8c564b',
                linewidth=2, label="C ↔ B (different noise)")
    ax.set_xlabel("Epoch (of model C)")
    ax.set_ylabel("Mean Max Cosine Similarity")
    ax.set_title("Pairwise: Same Init + Different Noise (C ↔ A/B)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()

    # Overlay plot
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.set_title(
        f"All Similarity Trajectories\n"
        f"(d_model={config['d_model']}, k={config['k']}, per_init={config['per_init']})",
        fontsize=13, fontweight='bold'
    )
    ax2.plot(epochs_a, logs_a["sim_to_init"], color=colors['A'], linewidth=2, linestyle='-', label="A → own init")
    ax2.plot(epochs_b, logs_b["sim_to_init"], color=colors['B'], linewidth=2, linestyle='-', label="B → own init")
    ax2.plot(epochs_c, logs_c["sim_to_init"], color=colors['C'], linewidth=2, linestyle='-', label="C → own init")
    ax2.plot(epochs_a, logs_a["sim_to_clean_centers"], color=colors['A'], linewidth=2, linestyle='--', label="A → clean centers")
    ax2.plot(epochs_b, logs_b["sim_to_clean_centers"], color=colors['B'], linewidth=2, linestyle='--', label="B → clean centers")
    ax2.plot(epochs_c, logs_c["sim_to_clean_centers"], color=colors['C'], linewidth=2, linestyle='--', label="C → clean centers")
    if "sim_to_A" in logs_b:
        ax2.plot(epochs_b, logs_b["sim_to_A"], color='#d62728', linewidth=2.5, linestyle=':', label="B ↔ A (same noise)")
    if "sim_to_A" in logs_c:
        ax2.plot(epochs_c, logs_c["sim_to_A"], color='#9467bd', linewidth=2.5, linestyle=':', label="C ↔ A (diff noise)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Mean Max Cosine Similarity", fontsize=12)
    ax2.legend(fontsize=9, ncol=2, loc='best')
    ax2.grid(True, alpha=0.3); ax2.set_ylim(bottom=0)

    overlay_path = output_path.replace('.pdf', '_overlay.pdf').replace('.png', '_overlay.png')
    if overlay_path == output_path:
        overlay_path = output_path + '_overlay.pdf'
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    print(f"Overlay figure saved to {overlay_path}")
    plt.close()

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, logs in [("A", logs_a), ("B", logs_b), ("C", logs_c)]:
        if not logs.get("sim_to_init"):
            continue
        print(f"\nModel {name}:")
        print(f"  Sim to init:    {logs['sim_to_init'][0]:.4f} → {logs['sim_to_init'][-1]:.4f}")
        print(f"  Sim to clean:   {logs['sim_to_clean_centers'][0]:.4f} → {logs['sim_to_clean_centers'][-1]:.4f}")
        print(f"  Final loss:     {logs['avg_loss'][-1]:.4f}")
    if "sim_to_A" in logs_b:
        print(f"\nPairwise (same noise):")
        print(f"  B↔A: {logs_b['sim_to_A'][0]:.4f} → {logs_b['sim_to_A'][-1]:.4f}")
    if "sim_to_A" in logs_c:
        print(f"\nPairwise (different noise):")
        print(f"  C↔A: {logs_c['sim_to_A'][0]:.4f} → {logs_c['sim_to_A'][-1]:.4f}")
    if "sim_to_B" in logs_c:
        print(f"  C↔B: {logs_c['sim_to_B'][0]:.4f} → {logs_c['sim_to_B'][-1]:.4f}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train SI-SAEs and track dictionary similarity over training"
    )
    parser.add_argument("shard_directory", type=str, nargs="?",
                        help="Directory containing shard_*.pt files")
    parser.add_argument("--plot-only", type=str, default=None, metavar="RESULTS_JSON",
                        help="Skip training; re-plot from an existing results JSON file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path for --plot-only (default: same dir as JSON)")
    parser.add_argument("--centers-dir", type=str, default="../centers",
                        help="Directory for pre-computed centers files")
    parser.add_argument("--output-dir", type=str, default="../similarity_results",
                        help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--d-model", type=int, default=5000,
                        help="Dictionary size (number of SAE concepts)")
    parser.add_argument("--k-fraction", type=float, default=0.01,
                        help="Sparsity fraction")
    parser.add_argument("--per-init", type=float, default=0.1,
                        help="SI noise level")
    parser.add_argument("--batch-size", type=int, default=16384,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use bfloat16 mixed precision (recommended for A100)")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Validation shard directory")
    parser.add_argument("--seed-a", type=int, default=42,
                        help="Init seed for models A and B (same noise)")
    parser.add_argument("--seed-c", type=int, default=43,
                        help="Init seed for model C (different noise)")
    parser.add_argument("--train-seed-a", type=int, default=100,
                        help="Training seed for model A")
    parser.add_argument("--train-seed-b", type=int, default=200,
                        help="Training seed for model B")
    parser.add_argument("--train-seed-c", type=int, default=300,
                        help="Training seed for model C")
    parser.add_argument("--dataset-size", type=int, default=None,
                        help="Override dataset size (for scheduler computation)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip figure generation after training")

    args = parser.parse_args()

    # ── Plot-only mode ────────────────────────────────────────────────────────
    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        out = args.output or os.path.join(os.path.dirname(args.plot_only), "similarity_study.pdf")
        plot_similarity_results(results, out)
        return

    if not args.shard_directory:
        parser.error("shard_directory is required unless --plot-only is specified")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Running on {torch.cuda.get_device_name(0)}, TF32 enabled")

    os.makedirs(args.output_dir, exist_ok=True)

    k = int(args.k_fraction * args.d_model)
    print(f"Config: d_model={args.d_model}, k={k}, per_init={args.per_init}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}")

    # ── Data loading ──
    shard_files = sorted(globmod.glob(os.path.join(args.shard_directory, 'shard_*.pt')))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files in {args.shard_directory}")
    print(f"Found {len(shard_files)} training shards")

    first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=True)
    d_brain = first_shard.shape[-1]
    samples_per_shard = first_shard.shape[0]
    dataset_size = args.dataset_size or len(shard_files) * samples_per_shard
    print(f"d_brain={d_brain}, estimated dataset_size={dataset_size}")

    raw_loader = create_dataloader(args.shard_directory, args.batch_size,
                                   num_workers=8, prefetch_factor=2)
    loader = DeviceDataLoader(raw_loader, device)

    val_loader = None
    if args.val_dir:
        raw_val = create_val_dataloader(args.val_dir, args.batch_size, num_workers=4, prefetch_factor=2)
        val_loader = DeviceDataLoader(raw_val, device)
        print(f"Validation enabled")

    # ── Load / compute clean k-means centers ──
    si_path = os.path.join(args.centers_dir, f"si_centers_{args.d_model}.pt")
    if os.path.exists(si_path):
        print(f"Loading cached centers from {si_path}")
        clean_centers = torch.load(si_path, map_location='cpu', weights_only=True)
    else:
        print("Computing k-means centers (this may take a while)...")
        clean_centers = cosine_kmeans(loader, args.d_model, d_brain)
        os.makedirs(args.centers_dir, exist_ok=True)
        torch.save(clean_centers, si_path)
    print(f"Clean centers shape: {clean_centers.shape}")

    total_steps = (dataset_size // args.batch_size) * args.epochs
    killer = GracefulKiller()

    # ══════════════════════════════════════════════════════════════════════════
    # Create 3 models:
    #   A: seed_a noise, train_seed_a training
    #   B: seed_a noise, train_seed_b training  (same init as A)
    #   C: seed_c noise, train_seed_c training  (different noise from A/B)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n── Creating Model A (seed_a noise) ──")
    sae_a, init_dict_a = create_si_sae(
        d_brain, args.d_model, k, device, clean_centers, args.per_init, args.seed_a)

    print("── Creating Model B (seed_a noise, same init as A) ──")
    sae_b, init_dict_b = create_si_sae(
        d_brain, args.d_model, k, device, clean_centers, args.per_init, args.seed_a)

    print("── Creating Model C (seed_c noise, different from A/B) ──")
    sae_c, init_dict_c = create_si_sae(
        d_brain, args.d_model, k, device, clean_centers, args.per_init, args.seed_c)

    ab_init_sim = mean_max_cosine_similarity(init_dict_a.to(device), init_dict_b.to(device))
    ac_init_sim = mean_max_cosine_similarity(init_dict_a.to(device), init_dict_c.to(device))
    print(f"\nInit similarity A↔B: {ab_init_sim:.6f} (should be ~1.0)")
    print(f"Init similarity A↔C: {ac_init_sim:.6f} (should be < 1.0)")

    # ── Train Model A ──
    torch.manual_seed(args.train_seed_a)
    torch.cuda.manual_seed_all(args.train_seed_a)
    opt_a, sched_a = create_optimizer_scheduler(sae_a, args.lr, total_steps)
    logs_a = train_with_similarity_tracking(
        sae_a, loader, opt_a, sched_a,
        nb_epochs=args.epochs, device=device, model_name="A",
        init_dict=init_dict_a, clean_centers=clean_centers,
        use_mixed_precision=args.mixed_precision,
        val_loader=val_loader,
    )
    if killer.kill_now:
        print("Interrupted after Model A. Saving partial results.")

    # ── Train Model B (tracking pairwise sim to A) ──
    if not killer.kill_now:
        torch.manual_seed(args.train_seed_b)
        torch.cuda.manual_seed_all(args.train_seed_b)
        opt_b, sched_b = create_optimizer_scheduler(sae_b, args.lr, total_steps)
        logs_b = train_with_similarity_tracking(
            sae_b, loader, opt_b, sched_b,
            nb_epochs=args.epochs, device=device, model_name="B",
            init_dict=init_dict_b, clean_centers=clean_centers,
            other_models={"A": sae_a},
            use_mixed_precision=args.mixed_precision,
            val_loader=val_loader,
        )
    else:
        logs_b = {}

    # ── Train Model C (tracking pairwise sim to A and B) ──
    if not killer.kill_now:
        torch.manual_seed(args.train_seed_c)
        torch.cuda.manual_seed_all(args.train_seed_c)
        opt_c, sched_c = create_optimizer_scheduler(sae_c, args.lr, total_steps)
        logs_c = train_with_similarity_tracking(
            sae_c, loader, opt_c, sched_c,
            nb_epochs=args.epochs, device=device, model_name="C",
            init_dict=init_dict_c, clean_centers=clean_centers,
            other_models={"A": sae_a, "B": sae_b},
            use_mixed_precision=args.mixed_precision,
            val_loader=val_loader,
        )
    else:
        logs_c = {}

    # ── Save results ──
    results = {
        "config": {
            "d_model": args.d_model,
            "k": k,
            "k_fraction": args.k_fraction,
            "per_init": args.per_init,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed_a": args.seed_a,
            "seed_c": args.seed_c,
            "train_seed_a": args.train_seed_a,
            "train_seed_b": args.train_seed_b,
            "train_seed_c": args.train_seed_c,
            "shard_directory": args.shard_directory,
        },
        "init_similarities": {
            "A_B": ab_init_sim,
            "A_C": ac_init_sim,
        },
        "model_A": logs_a,
        "model_B": logs_b,
        "model_C": logs_c,
    }

    out_path = os.path.join(args.output_dir, "similarity_study_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    for name, model in [("A", sae_a), ("B", sae_b), ("C", sae_c)]:
        model_path = os.path.join(args.output_dir, f"model_{name}_state_dict.pth")
        torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {args.output_dir}")

    # ── Generate figures ──
    if not args.no_plot:
        fig_path = os.path.join(args.output_dir, "similarity_study.pdf")
        plot_similarity_results(results, fig_path)


if __name__ == "__main__":
    main()
