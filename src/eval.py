import os
import json
import torch
import glob

# Local imports
from sae_factory import get_sae_model
from metric import evaluate_sae, compute_pairwise_stability, aggregate_metrics
from common import get_checkpoint_dir


def run_evaluation(config, args, device, loader, d_brain, run_suffix="", centers_dir="../centers"):
    """
    Load trained models and run evaluation metrics.

    Parameters
    ----------
    config : dict
        Training/eval configuration (d_model, k_fraction, num_saes, etc.).
    args : argparse.Namespace
        Must have: model_type, output_dir.  Optionally: single_sae.
    device : str or torch.device
    loader : DeviceDataLoader
    d_brain : int
        Input activation dimension.
    run_suffix : str
        Suffix used when saving models (e.g. "_d5000_k50").
    centers_dir : str
        Directory containing pre-computed centers files.

    Returns
    -------
    dict or None
        Final aggregated results, or None if no models were found.
    """
    model_type = args.model_type
    print(f"\n{'='*60}")
    print(f"--- Starting Evaluation for Model Type: {model_type} ---")
    print(f"{'='*60}")

    k = int(config['k_fraction'] * config['d_model'])
    all_dictionaries = []
    all_results = {}

    # Determine which SAEs to evaluate
    if getattr(args, 'single_sae', None):
        sae_indices = [args.single_sae]
    else:
        sae_indices = list(range(1, config['num_saes'] + 1))

    for i in sae_indices:
        model_id = f"sae_{i}_{model_type}{run_suffix}"
        save_path = os.path.join(args.output_dir, f"{model_id}_state_dict.pth")

        if not os.path.exists(save_path):
            print(f"[Warning] Trained model not found: {save_path}. Skipping SAE {i}.")
            continue

        print(f"\n--- Loading and Evaluating SAE {i}/{config['num_saes']} ---")

        sae = get_sae_model(
            model_type, d_brain, config['d_model'], k, device, loader, config,
            centers_dir=centers_dir,
        )

        sae.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        sae.to(device)
        sae.eval()

        print("Running evaluation...")
        with torch.inference_mode():
            metrics, dictionary = evaluate_sae(sae, loader, device)

        all_results[f"sae_{i}"] = metrics
        all_dictionaries.append(dictionary)
        print(f"SAE {i} Metrics: {json.dumps(metrics, indent=2)}")

    if not all_results:
        print(f"No completed SAEs found for {model_type}. Cannot aggregate results.")
        return None

    # --- Aggregate ---
    print("\n--- Aggregating Results ---")
    avg_metrics = aggregate_metrics(all_results)

    stability_metrics = compute_pairwise_stability(all_dictionaries, device)

    final_output = {
        "config": config,
        "model_type": model_type,
        "individual_averages": avg_metrics,
        "stability": stability_metrics,
        "raw": all_results
    }

    print("\n" + "="*60)
    print(f"FINAL RESULTS: {model_type}")
    print("="*60)
    print(json.dumps(final_output, indent=2))

    results_path = os.path.join(args.output_dir, f"final_results_{model_type}{run_suffix}_eval_only.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return final_output
