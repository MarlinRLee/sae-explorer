"""
Metrics for evaluating Sparse Autoencoders based on the Archetypal SAE paper.
Implements metrics across four dimensions:
1. Sparse Reconstruction (R^2, Dead Codes)
2. Consistency (Stability, Max Cosine, OOD Score)
3. Structure in Dictionary D (Stable Rank, Effective Rank, Coherence)
4. Structure in Codes Z (Connectivity, Negative Interference)
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations

Epsilon = 1e-6


# =============================================================================
# Dictionary-only metrics (don't need data iteration)
# =============================================================================

def stable_rank(D):
    """
    Compute the stable rank of the dictionary.
    
    Stable Rank = ||D||^2_F / ||D||^2_2
    """
    frobenius_norm_sq = (D ** 2).sum()
    spectral_norm_sq = torch.linalg.matrix_norm(D, ord=2) ** 2
    return (frobenius_norm_sq / (spectral_norm_sq + Epsilon)).item()


def effective_rank(D):
    """
    Compute the effective rank based on singular value entropy.
    
    Eff. Rank = exp(-sum_i std_i * log(std_i))
    """
    singular_values = torch.linalg.svdvals(D)
    singular_values = singular_values / (singular_values.sum() + Epsilon)
    singular_values = singular_values[singular_values > Epsilon]
    entropy = -(singular_values * torch.log(singular_values)).sum()
    return torch.exp(entropy).item()


def coherence(D, chunk_size=2048):
    """
    Compute the coherence (max non-diagonal cosine similarity).

    Coherence = max_{i≠j} |D_i^T D_j|

    Uses chunked computation to avoid OOM for large dictionaries (e.g. d=32000).
    """
    D_norm = D / (D.norm(dim=1, keepdim=True) + Epsilon)
    n = D_norm.shape[0]
    max_val = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = torch.matmul(D_norm[start:end], D_norm.T)  # (chunk, n)
        chunk.abs_()
        # Zero out the diagonal entries within this chunk
        diag_indices = torch.arange(start, end, device=D.device)
        local_indices = diag_indices - start
        chunk[local_indices, diag_indices] = 0.0
        max_val = max(max_val, chunk.max().item())
    return max_val


def stability(D1, D2):
    """
    Compute stability between two dictionaries using Hungarian algorithm.
    
    Stability(D, D') = max_{Π} (1/n) * Tr(D^T Π D')
    """
    D1_norm = D1 / (D1.norm(dim=1, keepdim=True) + Epsilon)
    D2_norm = D2 / (D2.norm(dim=1, keepdim=True) + Epsilon)
    
    similarity_matrix = torch.matmul(D1_norm, D2_norm.T)
    cost_matrix = -similarity_matrix.cpu().numpy()
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    aligned_similarities = similarity_matrix[row_ind, col_ind]
    
    return aligned_similarities.mean().item()


def max_cosine_similarity(D1, D2):
    """
    Compute maximum cosine similarity between any pair of atoms.
    
    Max Cosine = max_{i,j} <D_i, D'_j>
    """
    D1_norm = D1 / (D1.norm(dim=1, keepdim=True) + Epsilon)
    D2_norm = D2 / (D2.norm(dim=1, keepdim=True) + Epsilon)
    similarity_matrix = torch.matmul(D1_norm, D2_norm.T)
    return similarity_matrix.max().item()


# =============================================================================
# Streaming metric accumulators (memory efficient)
# =============================================================================

class StreamingR2:
    """
    Online computation of R² score using batched Welford's algorithm.
    
    Computes R² = 1 - SS_res / SS_tot where:
    - SS_res = sum of squared residuals  
    - SS_tot = total sum of squares (variance of x across samples, summed over dimensions)
    """
    def __init__(self):
        self.ss_res = 0.0  # Sum of squared residuals
        # For batched Welford's online variance algorithm
        self.n_samples = 0
        self.mean = None      # Running mean per dimension (d,)
        self.M2 = None        # Sum of squared deviations per dimension (d,)
    
    def update(self, x, x_hat):
        """
        x, x_hat: (batch_size, d)
        """
        batch_size = x.shape[0]
        
        # Accumulate residuals
        self.ss_res += ((x - x_hat) ** 2).sum().item()
        
        # Initialize accumulators on first batch
        if self.mean is None:
            self.mean = torch.zeros(x.shape[1], device=x.device, dtype=torch.float64)
            self.M2 = torch.zeros(x.shape[1], device=x.device, dtype=torch.float64)
        
        # Batched Welford update (Chan's parallel algorithm)
        x_f64 = x.double()
        batch_mean = x_f64.mean(dim=0)  # (d,)
        batch_var = x_f64.var(dim=0, unbiased=False)  # (d,)
        batch_M2 = batch_var * batch_size  # Sum of squared deviations for this batch
        
        # Combine with running statistics
        n_a = self.n_samples
        n_b = batch_size
        n_total = n_a + n_b
        
        if n_a == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            delta = batch_mean - self.mean
            self.mean = (n_a * self.mean + n_b * batch_mean) / n_total
            # Chan's formula for combining M2
            self.M2 = self.M2 + batch_M2 + delta**2 * n_a * n_b / n_total
        
        self.n_samples = n_total
    
    def compute(self):
        """
        Returns R² as a percentage (0-100 scale).
        """
        if self.n_samples < 2:
            return 0.0
        
        # SS_tot = sum of M2 across all dimensions
        ss_tot = self.M2.sum().item()
        
        r2 = 1.0 - (self.ss_res / (ss_tot + Epsilon))
        return r2 * 100  # Convert to percentage

class StreamingDeadCodes:
    """Track which codes have ever fired."""
    def __init__(self, num_codes, device):
        self.ever_fired = torch.zeros(num_codes, dtype=torch.bool, device=device)
    
    def update(self, z):
        # z: (batch, num_codes)
        batch_fired = (z.abs() > Epsilon).any(dim=0)
        self.ever_fired |= batch_fired
    
    def compute(self):
        return (~self.ever_fired).float().mean().item()


class StreamingOODScore:
    """Track max similarity of each dictionary atom to any data point."""
    def __init__(self, D):
        # D: (num_codes, dim)
        self.D_norm = D / (D.norm(dim=1, keepdim=True) + Epsilon)
        self.max_similarities = torch.zeros(D.shape[0], device=D.device)
    
    def update(self, x):
        # x: (batch, dim)
        x_norm = x / (x.norm(dim=1, keepdim=True) + Epsilon)
        # (num_codes, batch)
        similarities = torch.matmul(self.D_norm, x_norm.T)
        batch_max = similarities.max(dim=1).values
        self.max_similarities = torch.maximum(self.max_similarities, batch_max)
    
    def compute(self):
        return (1 - self.max_similarities.mean()).item()


class StreamingConnectivity:
    """Track co-activation patterns (which concept pairs ever co-activate)."""
    def __init__(self, num_codes, device):
        self.num_codes = num_codes
        # Track which pairs have co-activated
        self.coactivated = torch.zeros(num_codes, num_codes, dtype=torch.bool, device=device)
    
    def update(self, z):
        # z: (batch, num_codes)
        active = (z.abs() > Epsilon).float()  # (batch, num_codes)
        # Outer product for each sample, then OR across batch
        batch_coact = torch.matmul(active.T, active) > 0  # (num_codes, num_codes)
        self.coactivated |= batch_coact
    
    def compute(self):
        l0_norm = self.coactivated.float().sum()
        return (1 - l0_norm / (self.num_codes ** 2)).item()


class StreamingNegativeInterference:
    """Accumulate Z^T Z for negative interference computation."""
    def __init__(self, num_codes, device):
        self.ZTZ = torch.zeros(num_codes, num_codes, device=device)
        self.n_samples = 0

    def update(self, z):
        # z: (batch, num_codes)
        self.ZTZ += torch.matmul(z.T, z)
        self.n_samples += z.shape[0]

    def compute(self, D):
        ZTZ = self.ZTZ / max(self.n_samples, 1)
        DDT = torch.matmul(D, D.T)
        product = ZTZ * DDT
        neg_interference = torch.relu(-product)
        return neg_interference.norm().item()


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate_sae(sae, dataloader, device):
    """
    Comprehensive evaluation of an SAE computing all metrics from the A-SAE paper.
    Memory-efficient streaming implementation.
    
    Parameters
    ----------
    sae : nn.Module
        The SAE model. Expected interface:
        - pre_codes, codes, x_hat = sae(x)
        - D = sae.get_dictionary()
    dataloader : DataLoader
        DataLoader providing batches of activations.
    device : torch.device or str
        Device to run computations on.
    
    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    dictionary : torch.Tensor
        The dictionary D for stability computation.
    """
    sae.eval()
    
    # Get dictionary
    D = sae.get_dictionary().detach()  # (num_codes, dim)
    num_codes = D.shape[0]
    
    # Initialize streaming accumulators
    r2_acc = StreamingR2()
    dead_codes_acc = StreamingDeadCodes(num_codes, device)
    ood_acc = StreamingOODScore(D)
    connectivity_acc = StreamingConnectivity(num_codes, device)
    neg_inter_acc = StreamingNegativeInterference(num_codes, device)
    
    # Stream through data
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            if x.device != device:
                x = x.to(device)
            
            # Forward pass
            pre_codes, codes, x_hat = sae(x)
            
            # Update accumulators
            r2_acc.update(x, x_hat)
            dead_codes_acc.update(codes)
            ood_acc.update(x)
            connectivity_acc.update(codes)
            neg_inter_acc.update(codes)
    
    # Compute final metrics
    metrics = {
        # Sparse Reconstruction
        'R2': r2_acc.compute(),
        'Dead Codes': dead_codes_acc.compute(),
        
        # Consistency (OOD only - stability needs multiple SAEs)
        'OOD Score': ood_acc.compute(),
        
        # Dictionary Structure
        'Stable Rank': stable_rank(D),
        'Eff. Rank': effective_rank(D),
        'Coherence': coherence(D),
        
        # Codes Structure
        'Connectivity': connectivity_acc.compute(),
        'Neg. Inter.': neg_inter_acc.compute(D),
    }
    
    return metrics, D.cpu()


def aggregate_metrics(all_results):
    """
    Compute average metrics across multiple SAE results.

    Parameters
    ----------
    all_results : dict
        Mapping of SAE keys to metric dicts, e.g. {"sae_1": {...}, "sae_2": {...}}.

    Returns
    -------
    dict
        Averaged metrics with "avg_" prefix.
    """
    if not all_results:
        return {}
    avg_metrics = {}
    all_keys = set()
    for metrics in all_results.values():
        all_keys.update(metrics.keys())
    for key in all_keys:
        vals = [m[key] for m in all_results.values() if key in m]
        if vals:
            avg_metrics[f"avg_{key}"] = sum(vals) / len(vals)
    return avg_metrics


def compute_pairwise_stability(dictionaries, device):
    """
    Compute stability and max cosine metrics for all pairs of dictionaries.
    
    Parameters
    ----------
    dictionaries : list of torch.Tensor
        List of dictionary tensors, each of shape (num_codes, dim).
    device : torch.device or str
        Device for computation.
    
    Returns
    -------
    dict
        Contains 'avg_stability', 'avg_max_cosine', and per-pair values.
    """
    n = len(dictionaries)
    if n < 2:
        return {'avg_stability': None, 'avg_max_cosine': None, 'pairs': {}}
    
    stabilities = []
    max_cosines = []
    pair_results = {}
    
    for (i, D1), (j, D2) in combinations(enumerate(dictionaries), 2):
        D1_dev = D1.to(device)
        D2_dev = D2.to(device)
        
        stab = stability(D1_dev, D2_dev)
        max_cos = max_cosine_similarity(D1_dev, D2_dev)
        
        stabilities.append(stab)
        max_cosines.append(max_cos)
        pair_results[f"sae_{i+1}_vs_sae_{j+1}"] = {
            'stability': stab,
            'max_cosine': max_cos
        }
    
    return {
        'avg_stability': sum(stabilities) / len(stabilities),
        'avg_max_cosine': sum(max_cosines) / len(max_cosines),
        'pairs': pair_results
    }