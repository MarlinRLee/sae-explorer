import os
import torch
import torch.nn as nn
from overcomplete.sae import TopKSAE, RATopKSAE
from utils import cosine_kmeans

def get_sae_model(model_type, d_brain, d_model, k, device, dataloader, config, centers_dir="../centers", n_centers=None):
    """
    Factory to build or load the correct SAE variant.
    Handles the specific initialization logic for RA-SAE and SI-SAE.

    Parameters
    ----------
    n_centers : int or None
        Number of K-means centers to use for SI-SAE initialization.
        If None, defaults to d_model (all atoms from K-means).
        If < d_model, the remaining (d_model - n_centers) atoms are pure noise.
    """
    if model_type == "RA-SAE":
        # RA-SAE requires pre-computed centers
        centers_path = os.path.join(centers_dir, f"centers_{config['n_candidates']}.pt")
        if os.path.exists(centers_path):
            print(f"Loading cached centers from {centers_path}")
            c_tensor = torch.load(centers_path, map_location=device, weights_only=True)
        else:
            c_tensor = cosine_kmeans(dataloader, config['n_candidates'], d_brain)
            torch.save(c_tensor, centers_path)
            
        sae = RATopKSAE(d_brain, nb_concepts=d_model, top_k=k, points=c_tensor.float().to(device),
                delta=config['delta'], device=device)
        return sae

    sae = TopKSAE(input_shape=d_brain, 
                  nb_concepts=d_model, 
                  top_k=k, 
                  device=device)
    
    if model_type == "SAE":
        return sae
    elif model_type == "SI-SAE":
        # SI-SAE: Initialize Standard SAE with Noisy K-Means Centroids
        # n_centers controls how many atoms come from K-means vs pure noise
        if n_centers is None:
            n_centers = d_model

        n_centers = min(n_centers, d_model)

        si_path = os.path.join(centers_dir, f"si_centers_{n_centers}.pt")
        if os.path.exists(si_path):
            si_centers = torch.load(si_path, map_location=device, weights_only=True)
        else:
            si_centers = cosine_kmeans(dataloader, n_centers, d_brain)
            os.makedirs(centers_dir, exist_ok=True)
            torch.save(si_centers, si_path)

        # Apply SI initialization to the K-means portion
        centers = si_centers.cpu()
        noise = torch.randn_like(centers) * centers.std() * config['per_init']
        kmeans_weights = (1 - config['per_init']) * centers + noise

        if n_centers < d_model:
            # Remaining atoms are pure random noise (matched scale)
            random_weights = torch.randn(d_model - n_centers, centers.shape[-1])
            weights = torch.cat([kmeans_weights, random_weights], dim=0)
            print(f"SI-SAE init: {n_centers} K-means centers + {d_model - n_centers} random atoms")
        else:
            weights = kmeans_weights

        # Set Dictionary and Encoder
        norm_weights = torch.nn.functional.normalize(weights.to(device), p=2, dim=-1)
        sae.dictionary._weights.data = norm_weights

        enc_weights = norm_weights.clone() * (1.0 / (k ** 0.5))
        sae.encoder.final_block[0].weight.data = enc_weights

        return sae