import os
import json
import argparse
import torch
import glob

# Local imports
from data import create_dataloader, DeviceDataLoader, create_val_dataloader
from sae_factory import get_sae_model
from metric import evaluate_sae, compute_pairwise_stability, aggregate_metrics
from train import train_sae
from common import GracefulKiller, get_checkpoint_dir, is_training_complete, criterion, create_optimizer_scheduler
from eval import run_evaluation


# --- Configuration ---
CONFIG = {
    'num_saes': 1,
    'd_model': 32_000,
    'k_fraction': 0.0025,
    'epochs': 30,
    'batch_size': 16_384,
    'prefetch_factor': 2,
    'num_workers': 8,
    'lr': 5e-4,
    # Approximate number of samples in the training dataset
    'dataset_size': 250_000_000,
    #RA SAE:
    'n_candidates': 32_000,
    'delta': 1.0,
    #SI SAE
    'per_init': 0.1,

    # Reanimation / dead feature recovery
    'reanimate_coeff': 0.33,
    'resample_every_n_epochs': 0,
    'dead_threshold': 1e-6,

    #Meta or validation
    'checkpoint_every_n_epochs': 5,
    # Validation and early stopping settings
    'val_batch_size': 16_384,  # Batch size for validation
    'val_num_workers': 4,      # Workers for validation loader
    'val_subset_fraction': 1.0, # Use all validation data (or reduce for speed)
    'early_stopping_patience': 10,  # Stop if no improvement for N epochs
    'early_stopping_min_delta': 0,  # Minimum improvement to count
}


def build_run_suffix(model_type, config, n_centers=None):
    """Build a descriptive suffix for checkpoint/output file naming."""
    k = int(config['k_fraction'] * config['d_model'])
    run_suffix = f"_d{config['d_model']}_k{k}"
    if model_type == "RA-SAE":
        run_suffix += f"_delta{config['delta']}"
    elif model_type == "SI-SAE":
        run_suffix += f"_per_init{config['per_init']}"
        if n_centers is not None and n_centers != config['d_model']:
            run_suffix += f"_nc{n_centers}"
    return run_suffix


def main():
    parser = argparse.ArgumentParser(description="Train SAE with checkpointing support")
    parser.add_argument("shard_directory", type=str, help="Directory containing shard_*.pt files")
    parser.add_argument("model_type", type=str, choices=["SAE", "RA-SAE", "SI-SAE"])
    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints",
                        help="Base directory for checkpoints")
    parser.add_argument("--output-dir", type=str, default="../trained_models",
                        help="Directory for final trained models")
    parser.add_argument("--centers-dir", type=str, default="../centers",
                        help="Directory for pre-computed centers files")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N epochs")

    parser.add_argument("--d-model", type=int, default=None,
                        help="Override d_model (number of SAE features)")
    parser.add_argument("--k-fraction", type=float, default=None,
                        help="Override k_fraction (e.g., 0.01, 0.05, 0.1)")
    parser.add_argument("--per-init", type=float, default=None,
                        help="Override per_init for SI-SAE (e.g., 0, 0.01, 0.1, 0.4, 0.8)")
    parser.add_argument("--n-centers", type=int, default=None,
                        help="Number of K-means centers for SI-SAE (default: d_model). "
                             "Remaining atoms initialized as pure noise.")

    # Eval-only mode
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved models and run evaluation only.")

    # Validation arguments
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Directory containing validation shard_*.pt files. "
                             "If not provided, validation is disabled.")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping even if validation is enabled")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Override early stopping patience")
    parser.add_argument("--val-subset", type=float, default=None,
                        help="Fraction of validation shards to use (0.0-1.0)")

    # Reanimation arguments
    parser.add_argument("--reanimate-coeff", type=float, default=None,
                        help="Auxiliary reanimation loss weight (default: 0.33)")
    parser.add_argument("--resample-every", type=int, default=None,
                        help="Resample dead features every N epochs (0=disabled)")
    parser.add_argument("--dead-threshold", type=float, default=None,
                        help="Frequency below which a feature is considered dead")
    parser.add_argument("--no-reanimate", action="store_true",
                        help="Disable reanimation entirely (sets coeff to 0)")

    # Performance arguments
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use bfloat16 mixed precision training (recommended for A100)")

    args = parser.parse_args()

    if args.epochs is not None:
        CONFIG['epochs'] = args.epochs
    if args.checkpoint_every is not None:
        CONFIG['checkpoint_every_n_epochs'] = args.checkpoint_every
    if args.early_stopping_patience is not None:
        CONFIG['early_stopping_patience'] = args.early_stopping_patience
    if args.val_subset is not None:
        CONFIG['val_subset_fraction'] = args.val_subset

    if args.d_model is not None:
        CONFIG['d_model'] = args.d_model
    if args.k_fraction is not None:
        CONFIG['k_fraction'] = args.k_fraction
    if args.per_init is not None:
        CONFIG['per_init'] = args.per_init

    # Reanimation overrides
    if args.no_reanimate:
        CONFIG['reanimate_coeff'] = 0.0
        CONFIG['resample_every_n_epochs'] = 0
    else:
        if args.reanimate_coeff is not None:
            CONFIG['reanimate_coeff'] = args.reanimate_coeff
        if args.resample_every is not None:
            CONFIG['resample_every_n_epochs'] = args.resample_every
    if args.dead_threshold is not None:
        CONFIG['dead_threshold'] = args.dead_threshold

    k = int(CONFIG['k_fraction'] * CONFIG['d_model'])
    print(f"k_fraction: {CONFIG['k_fraction']} (k={k})")
    print(f"per_init: {CONFIG['per_init']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enable TF32 for free speedup on Ampere+ GPUs (A100, etc.)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for matrix multiplications")

    print(f"Running {args.model_type} on {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    n_centers = args.n_centers  # None means use d_model (all K-means)
    run_suffix = build_run_suffix(args.model_type, CONFIG, n_centers=n_centers)

    # Update checkpoint directory call to use the suffix
    checkpoint_dir = get_checkpoint_dir(args.checkpoint_dir, args.model_type, run_suffix)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")

    # --- Training Data ---
    shard_files = sorted(glob.glob(os.path.join(args.shard_directory, 'shard_*.pt')))
    if not shard_files:
        raise FileNotFoundError(f"No .pt shards found in {args.shard_directory}")

    print(f"Found {len(shard_files)} training shard files")

    first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=True)
    d_brain = first_shard.shape[-1]
    samples_per_shard = first_shard.shape[0]
    CONFIG['dataset_size'] = len(shard_files) * samples_per_shard
    print(f"Detected embedding dimension: {d_brain}")
    print(f"Estimated dataset size: {CONFIG['dataset_size']} ({len(shard_files)} shards x {samples_per_shard} samples)")

    raw_loader = create_dataloader(
        args.shard_directory,
        CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        prefetch_factor=CONFIG['prefetch_factor']
    )
    loader = DeviceDataLoader(raw_loader, device)

    # --- Eval-only mode ---
    if args.eval_only:
        print("\n** Eval-only mode **")
        run_evaluation(
            CONFIG, args, device, loader, d_brain,
            run_suffix=run_suffix, centers_dir=args.centers_dir,
        )
        print("\nDone (eval-only).")
        return

    # --- Validation Data ---
    val_loader = None
    early_stopping_patience = None

    if args.val_dir:
        val_shard_files = sorted(glob.glob(os.path.join(args.val_dir, 'shard_*.pt')))
        if not val_shard_files:
            print(f"[Warning] No validation shards found in {args.val_dir}. Disabling validation.")
        else:
            print(f"Found {len(val_shard_files)} validation shard files")

            raw_val_loader = create_val_dataloader(
                args.val_dir,
                CONFIG['val_batch_size'],
                num_workers=CONFIG['val_num_workers'],
                prefetch_factor=2,
                subset_fraction=CONFIG['val_subset_fraction']
            )
            val_loader = DeviceDataLoader(raw_val_loader, device)
            print(f"Validation enabled with {len(val_shard_files)} shards "
                  f"(using {CONFIG['val_subset_fraction']*100:.0f}%)")

            # Enable early stopping unless explicitly disabled
            if not args.no_early_stopping:
                early_stopping_patience = CONFIG['early_stopping_patience']
                print(f"Early stopping enabled (patience={early_stopping_patience})")
            else:
                print("Early stopping disabled by user")
    else:
        print("No validation directory provided. Training without validation.")

    # --- Training ---
    killer = GracefulKiller()

    print(f"Config: d_model={CONFIG['d_model']}, k={k}, epochs={CONFIG['epochs']}")

    all_dictionaries = []
    all_results = {}

    sae_indices = list(range(1, CONFIG['num_saes'] + 1))
    for i in sae_indices:
        if killer.kill_now:
            print("[Signal] Stopping before next SAE due to shutdown signal")
            break

        print(f"\n{'='*60}")
        print(f"--- SAE {i}/{CONFIG['num_saes']} ---")
        print(f"{'='*60}")

        model_id = f"sae_{i}_{args.model_type}{run_suffix}"
        save_path = os.path.join(args.output_dir, f"{model_id}_state_dict.pth")

        if os.path.exists(save_path) and is_training_complete(checkpoint_dir, i, CONFIG['epochs']):
            print(f"Loading existing completed model from {save_path}")
            sae = get_sae_model(
                args.model_type, d_brain, CONFIG['d_model'], k, device, loader, CONFIG,
                centers_dir=args.centers_dir, n_centers=n_centers,
            )
            sae.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        else:
            print("Training (will resume from checkpoint if available)...")

            sae = get_sae_model(
                args.model_type, d_brain, CONFIG['d_model'], k, device, loader, CONFIG,
                centers_dir=args.centers_dir, n_centers=n_centers,
            )

            total_steps = (CONFIG['dataset_size'] // CONFIG['batch_size']) * CONFIG['epochs']
            optimizer, scheduler = create_optimizer_scheduler(sae, CONFIG['lr'], total_steps)

            logs = train_sae(
                sae, loader, criterion, optimizer,
                scheduler=scheduler,
                nb_epochs=CONFIG['epochs'],
                device=device,
                sae_index=i,
                monitoring=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every_n_epochs=CONFIG['checkpoint_every_n_epochs'],
                model_type=args.model_type,
                # Validation parameters
                val_loader=val_loader,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=CONFIG['early_stopping_min_delta'],
                # Performance
                use_mixed_precision=args.mixed_precision,
                # Reanimation
                reanimate_coeff=CONFIG['reanimate_coeff'],
                resample_every_n_epochs=CONFIG['resample_every_n_epochs'],
                dead_threshold=CONFIG['dead_threshold'],
            )

            torch.save(sae.state_dict(), save_path)
            print(f"Saved final model to {save_path}")


        print("Running evaluation on training data...")
        with torch.inference_mode():
            metrics, dictionary = evaluate_sae(sae, loader, device)

        if val_loader is not None:
            print("Running evaluation on validation data...")
            with torch.inference_mode():
                val_metrics, _ = evaluate_sae(sae, val_loader, device)
            for key, val in val_metrics.items():
                metrics[f"val_{key}"] = val

        all_results[f"sae_{i}"] = metrics
        all_dictionaries.append(dictionary)
        print(f"SAE {i} Metrics: {json.dumps(metrics, indent=2)}")

    if len(all_results) < CONFIG['num_saes']:
        print(f"\n[Warning] Only completed {len(all_results)}/{CONFIG['num_saes']} SAEs")
        if killer.kill_now:
            print("Training was interrupted. Run again to continue.")

        partial_output = {
            "CONFIG": CONFIG,
            "completed_saes": list(all_results.keys()),
            "raw": all_results
        }
        partial_path = os.path.join(args.output_dir, f"partial_results_{args.model_type}{run_suffix}.json")
        with open(partial_path, "w") as f:
            json.dump(partial_output, f, indent=2)
        print(f"Saved partial results to {partial_path}")
        return

    print("\n--- Aggregating Results ---")
    avg_metrics = aggregate_metrics(all_results)

    stability_metrics = compute_pairwise_stability(all_dictionaries, device)

    final_output = {
        "CONFIG": CONFIG,
        "individual_averages": avg_metrics,
        "stability": stability_metrics,
        "raw": all_results
    }

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(final_output, indent=2))

    results_path = os.path.join(args.output_dir, f"final_results_{args.model_type}{run_suffix}.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
