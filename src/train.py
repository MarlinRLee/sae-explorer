import time
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from einops import rearrange

from overcomplete.metrics import l2, r2_score, l0_eps
from overcomplete.sae.trackers import DeadCodeTracker
from overcomplete.sae.train import extract_input


def reanimation_loss(model, x, x_hat, dead_mask, top_k_aux=None):
    """
    Auxiliary loss to reanimate dead features.

    Dead features are encouraged to reconstruct the residual (x - x_hat)
    that the live features failed to capture. This follows the approach from
    recent SAE work (Anthropic, OpenAI).

    Parameters
    ----------
    model : TopKSAE
        The SAE model (needs encoder weights and dictionary).
    x : Tensor (batch, d_brain)
        Original input.
    x_hat : Tensor (batch, d_brain)
        Reconstruction from the main forward pass.
    dead_mask : Tensor (n_features,) bool
        True for dead features.
    top_k_aux : int or None
        Number of dead features to activate per sample.
        Defaults to the model's top_k.

    Returns
    -------
    Tensor (scalar)
        Auxiliary MSE loss on the residual, or 0 if no dead features.
    """
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return torch.tensor(0.0, device=x.device)

    if top_k_aux is None:
        top_k_aux = model.top_k

    # Clamp to number of dead features available
    top_k_aux = min(top_k_aux, n_dead)

    residual = (x - x_hat).detach()  # Don't backprop through live features

    # Encode residual through dead features only
    encoder_weight = model.encoder.final_block[0].weight  # (n_features, d_brain)
    dead_enc_weight = encoder_weight[dead_mask]            # (n_dead, d_brain)
    z_dead = F.relu(residual @ dead_enc_weight.T)          # (batch, n_dead)

    # Apply top-k sparsity to dead features
    if n_dead > top_k_aux:
        topk = torch.topk(z_dead, top_k_aux, dim=-1)
        z_dead_sparse = torch.zeros_like(z_dead)
        z_dead_sparse.scatter_(-1, topk.indices, topk.values)
        z_dead = z_dead_sparse

    # Decode through dead dictionary atoms
    D = model.get_dictionary()
    dead_dict = D[dead_mask]                               # (n_dead, d_brain)
    residual_hat = z_dead @ dead_dict                      # (batch, d_brain)

    # MSE on residual reconstruction
    return (residual - residual_hat).square().mean()


def resample_dead_features(model, dataloader, dead_mask, optimizer, device, num_batches=5):
    """
    Reinitialize persistently dead features using high-loss input samples.

    Parameters
    ----------
    model : SAE
        The SAE model.
    dataloader : DataLoader
        Training data loader (used to collect seed samples).
    dead_mask : Tensor (n_features,) bool
        True for dead features to resample.
    optimizer : torch.optim.Optimizer
        Optimizer whose state needs resetting for resampled params.
    device : str
        Device.
    num_batches : int
        Number of batches to collect for computing per-sample loss.
    """
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    model.eval()
    all_x = []
    all_losses = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            x = extract_input(batch).float()
            _, _, x_hat = model(x)
            # Per-sample reconstruction loss
            per_sample_loss = (x - x_hat.float()).square().mean(dim=-1)  # (batch,)
            all_x.append(x)
            all_losses.append(per_sample_loss)

    model.train()

    all_x = torch.cat(all_x, dim=0)           # (N, d_brain)
    all_losses = torch.cat(all_losses, dim=0)  # (N,)

    # Select top-loss samples as seeds for dead features
    n_seeds = min(n_dead, all_x.shape[0])
    top_indices = torch.topk(all_losses, n_seeds).indices
    seeds = all_x[top_indices]  # (n_seeds, d_brain)

    # L2-normalize seeds
    seeds = F.normalize(seeds, dim=-1)

    # Reinitialize dead encoder rows
    encoder_weight = model.encoder.final_block[0].weight  # (n_features, d_brain)
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]

    with torch.no_grad():
        for j, idx in enumerate(dead_indices[:n_seeds]):
            encoder_weight[idx] = seeds[j]

        # Reinitialize dead dictionary rows
        dict_weight = model.dictionary._weights  # actual parameter (nn.Parameter)
        for j, idx in enumerate(dead_indices[:n_seeds]):
            dict_weight[idx] = seeds[j]

        # Reset optimizer state for affected parameters
        for param in [encoder_weight, dict_weight]:
            if param in optimizer.state:
                state = optimizer.state[param]
                for key in ['exp_avg', 'exp_avg_sq']:
                    if key in state:
                        state[key][dead_indices[:n_seeds]] = 0.0

    return n_seeds


def _compute_reconstruction_error(x, x_hat):
    """
    Try to match the shapes of x and x_hat to compute the reconstruction error.
    
    Ensures both tensors are float32 for consistent computation.
    """
    # Ensure consistent dtype for accurate comparison
    x = x.float()
    x_hat = x_hat.float()
    
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n c w h -> (n w h) c')
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n t c -> (n t) c')
    else:
        assert x.shape == x_hat.shape, "Input and output shapes must match."
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)
    return r2.item()


def _log_metrics(monitoring, logs, model, z, loss, optimizer, current_step, sae_prefix):
    """Log training metrics for the current training step."""
    if monitoring == 0:
        return

    if monitoring > 0:
        lr = optimizer.param_groups[0]['lr']
        step_loss = loss.item()
        logs['lr'].append(lr)
        logs['step_loss'].append(step_loss)

    if monitoring > 1:
        z_l2 = l2(z.float()).item()
        dictionary_sparsity = l0_eps(model.get_dictionary()).mean().item()
        dictionary_norms = l2(model.get_dictionary(), -1).mean().item()
        
        logs['z_l2'].append(z_l2)
        logs['dictionary_sparsity'].append(dictionary_sparsity)
        logs['dictionary_norms'].append(dictionary_norms)


def save_checkpoint(checkpoint_dir, sae_index, epoch, global_step, model, optimizer, 
                    scheduler, logs, best_loss=None, early_stopping_state=None):
    """Save a training checkpoint for resuming later."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'logs': dict(logs),
        'best_loss': best_loss,
        'training_complete': False,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if early_stopping_state is not None:
        checkpoint['early_stopping_state'] = early_stopping_state
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_sae_{sae_index}.pt')
    temp_path = checkpoint_path + '.tmp'
    
    # Save to temp file first, then rename (atomic operation)
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)
    
    print(f"  [Checkpoint] Saved at epoch {epoch+1}, step {global_step}")
    return checkpoint_path


def load_checkpoint(checkpoint_dir, sae_index, model, optimizer, scheduler=None, device='cuda'):
    """Load a training checkpoint if it exists."""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_sae_{sae_index}.pt')
    
    if not os.path.exists(checkpoint_path):
        return None
    
    print(f"  [Checkpoint] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logs = defaultdict(list, checkpoint.get('logs', {}))
    
    print(f"  [Checkpoint] Resuming from epoch {checkpoint['epoch']+1}, step {checkpoint['global_step']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'logs': logs,
        'best_loss': checkpoint.get('best_loss'),
        'early_stopping_state': checkpoint.get('early_stopping_state'),
    }


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum change in validation loss to qualify as an improvement.
    mode : str
        'min' for loss (lower is better), 'max' for metrics like R2 (higher is better).
    """
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """
        Check if training should stop.
        
        Returns
        -------
        bool
            True if this is a new best score, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def get_state(self):
        """Get state for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch,
        }
    
    def load_state(self, state):
        """Load state from checkpoint."""
        if state is not None:
            self.counter = state['counter']
            self.best_score = state['best_score']
            self.early_stop = state['early_stop']
            self.best_epoch = state['best_epoch']


def validate(model, val_loader, criterion, device, omnipresent_threshold=0.9):
    """Run validation and calculate feature density/omnipresence."""
    model.eval()

    total_loss = 0.0
    total_r2 = 0.0
    total_samples = 0
    batch_count = 0

    # Infer dictionary size from the model
    d_sae = model.get_dictionary().shape[0]
    feature_acts_count = torch.zeros(d_sae, device=device)

    with torch.no_grad():
        for batch in val_loader:
            x = extract_input(batch)  # already on device via DeviceDataLoader
            x = x.float()
            
            z_pre, z, x_hat = model(x)
            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
            
            # Update metrics
            total_loss += loss.item()
            total_r2 += _compute_reconstruction_error(x, x_hat)
            total_samples += x.shape[0]
            batch_count += 1
            
            feature_acts_count += (z > 0).float().sum(dim=0)
    
    # Calculate frequency per feature
    feature_freqs = feature_acts_count / total_samples
    
    # Calculate omnipresent percentage
    omnipresent_mask = feature_freqs >= omnipresent_threshold
    pct_omnipresent = omnipresent_mask.float().mean().item() * 100
    
    # Calculate dead features (never fire) for comparison
    pct_dead = (feature_freqs == 0).float().mean().item() * 100

    model.train()
    
    return {
        'val_loss': total_loss / batch_count if batch_count > 0 else float('inf'),
        'val_r2': total_r2 / batch_count if batch_count > 0 else 0.0,
        'val_omnipresent': pct_omnipresent,
        'val_dead': pct_dead
    }

def train_sae(model, dataloader, criterion, optimizer, scheduler=None,
              nb_epochs=20, clip_grad=1.0, monitoring=1, device="cpu", sae_index=1,
              checkpoint_dir=None, checkpoint_every_n_epochs=1, model_type=None,
              val_loader=None, early_stopping_patience=None, early_stopping_min_delta=0.0,
              use_mixed_precision=False,
              reanimate_coeff=0.0, resample_every_n_epochs=0,
              dead_threshold=1e-6, reanimate_after_epoch=2):
    """
    Train a Sparse Autoencoder (SAE) model with checkpointing support.
    
    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        Training data loader.
    criterion : callable
        Loss function.
    optimizer : Optimizer
        Optimizer for training.
    scheduler : LRScheduler, optional
        Learning rate scheduler.
    nb_epochs : int
        Maximum number of epochs to train.
    clip_grad : float
        Gradient clipping value.
    monitoring : int
        Level of monitoring (0=none, 1=basic, 2=detailed).
    device : str
        Device to train on.
    sae_index : int
        Index of the SAE (for checkpointing).
    checkpoint_dir : str, optional
        Directory to save checkpoints.
    checkpoint_every_n_epochs : int
        Save checkpoint every N epochs.
    model_type : str, optional
        Type of model (used for SI-SAE freezing).
    val_loader : DataLoader, optional
        Validation data loader. If provided, validation will be run each epoch.
    early_stopping_patience : int, optional
        If provided, enables early stopping with this patience value.
    early_stopping_min_delta : float
        Minimum improvement required for early stopping.
    use_mixed_precision : bool
        If True, use bfloat16 mixed precision training (recommended for A100).
    reanimate_coeff : float
        Weight for auxiliary reanimation loss (0 = disabled).
    resample_every_n_epochs : int
        Weight resampling interval in epochs (0 = disabled).
    dead_threshold : float
        Frequency below which a feature is considered dead.
    reanimate_after_epoch : int
        Don't reanimate before this epoch (aligns with SI-SAE freeze period).

    Returns
    -------
    dict
        Training logs.
    """
    logs = defaultdict(list)
    global_step = 0
    start_epoch = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    sae_prefix = f"sae_{sae_index}"
    
    # Mixed precision setup
    if use_mixed_precision:
        if not torch.cuda.is_available():
            print("  [Warning] Mixed precision requested but CUDA not available. Disabling.")
            use_mixed_precision = False
        elif not torch.cuda.is_bf16_supported():
            print("  [Warning] BF16 not supported on this GPU. Disabling mixed precision.")
            use_mixed_precision = False
        else:
            print("  [Mixed Precision] Using bfloat16 (no loss scaling needed)")
    
    # Initialize early stopping if requested
    early_stopper = None
    if early_stopping_patience is not None and val_loader is not None:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='min'  # We're tracking validation loss
        )
    
    # Try to resume from checkpoint
    if checkpoint_dir:
        resumed = load_checkpoint(checkpoint_dir, sae_index, model, optimizer, scheduler, device=device)
        if resumed:
            start_epoch = resumed['epoch'] + 1
            global_step = resumed['global_step']
            logs = resumed['logs']
            best_loss = resumed.get('best_loss', float('inf'))
            
            # Restore early stopping state
            if early_stopper is not None and resumed.get('early_stopping_state'):
                early_stopper.load_state(resumed['early_stopping_state'])
                best_val_loss = early_stopper.best_score if early_stopper.best_score else float('inf')
            
            if start_epoch >= nb_epochs:
                print(f"  [Checkpoint] Training already complete for {sae_prefix}")
                return logs

    print(f"Starting training for {sae_prefix} from epoch {start_epoch + 1}")
    print(f"  Mixed precision: {'bfloat16' if use_mixed_precision else 'disabled (fp32)'}")
    
    if val_loader is not None:
        print(f"  Validation enabled")
        if early_stopper is not None:
            print(f"  Early stopping enabled (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})")

    if reanimate_coeff > 0:
        print(f"  Reanimation aux loss enabled (coeff={reanimate_coeff}, after epoch {reanimate_after_epoch})")
    if resample_every_n_epochs > 0:
        print(f"  Weight resampling enabled (every {resample_every_n_epochs} epochs, after epoch {reanimate_after_epoch})")

    frozen = False
    if model_type == "SI-SAE" and start_epoch < 2:
        for param in model.dictionary.parameters():
            param.requires_grad = False
        frozen = True
            
            
    for epoch in range(start_epoch, nb_epochs):
        if frozen and epoch >= 2:
            for param in model.dictionary.parameters():
                param.requires_grad = True
            print("unfreeze dict", flush = True)
            frozen = False
            
        model.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0
        mon_count = 0
        dead_tracker = None

        for batch in dataloader:
            global_step += 1
            batch_count += 1
            
            x = extract_input(batch)  # already on device via DeviceDataLoader
            
            optimizer.zero_grad(set_to_none=True)

            if use_mixed_precision:
                # === MIXED PRECISION TRAINING ===
                # Forward pass in bfloat16
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    z_pre, z, x_hat = model(x)
                
                # Compute loss in fp32 for numerical stability
                # This is the key to avoiding NaN/Inf issues
                loss = criterion(
                    x.float(), 
                    x_hat.float(), 
                    z_pre.float(), 
                    z.float(), 
                    model.get_dictionary().float()
                )
                
                # Dead tracker needs float
                if dead_tracker is None:
                    dead_tracker = DeadCodeTracker(z.shape[1], device)
                dead_tracker.update(z.float())

                # Auxiliary reanimation loss for dead features
                if reanimate_coeff > 0 and epoch >= reanimate_after_epoch:
                    dead_mask = ~dead_tracker.alive_features
                    if dead_mask.any():
                        aux_loss = reanimation_loss(model, x.float(), x_hat.float(), dead_mask)
                        loss = loss + reanimate_coeff * aux_loss

                # Backward pass - gradients computed in mixed precision
                # but accumulated in fp32 (PyTorch handles this automatically)
                loss.backward()
                
            else:
                # === STANDARD FP32 TRAINING ===
                x = x.float()
                
                z_pre, z, x_hat = model(x)
                loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())

                if dead_tracker is None:
                    dead_tracker = DeadCodeTracker(z.shape[1], device)
                dead_tracker.update(z)

                # Auxiliary reanimation loss for dead features
                if reanimate_coeff > 0 and epoch >= reanimate_after_epoch:
                    dead_mask = ~dead_tracker.alive_features
                    if dead_mask.any():
                        aux_loss = reanimation_loss(model, x, x_hat, dead_mask)
                        loss = loss + reanimate_coeff * aux_loss

                loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if monitoring and batch_count % 50 == 0:
                mon_count += 1
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z.float(), 0).sum().item()
                _log_metrics(monitoring, logs, model, z.float(), loss, optimizer, global_step, sae_prefix) 

        # Weight resampling for dead features at epoch boundaries
        if (resample_every_n_epochs > 0
                and epoch >= reanimate_after_epoch
                and (epoch - reanimate_after_epoch) % resample_every_n_epochs == 0
                and dead_tracker is not None):
            dead_mask = ~dead_tracker.alive_features
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                n_resampled = resample_dead_features(
                    model, dataloader, dead_mask, optimizer, device
                )
                print(f"  [Resample] Epoch {epoch+1}: resampled {n_resampled}/{n_dead} dead features")
                logs['resampled_features'].append(n_resampled)
            else:
                logs['resampled_features'].append(0)

        epoch_duration = time.time() - start_time

        # Training metrics
        if monitoring and batch_count > 0 and mon_count > 0:
            avg_loss = epoch_loss / mon_count
            avg_error = epoch_error / mon_count
            avg_sparsity = epoch_sparsity / mon_count
            dead_ratio = dead_tracker.get_dead_ratio()

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['time_epoch'].append(epoch_duration)
            logs['z_sparsity'].append(avg_sparsity)
            logs['dead_features'].append(dead_ratio)
            
            if avg_loss < best_loss:
                best_loss = avg_loss

            train_msg = (f"Epoch[{epoch+1}/{nb_epochs}] Train - Loss: {avg_loss:.4f}, "
                        f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                        f"Dead: {dead_ratio*100:.1f}%, Time: {epoch_duration:.2f}s")
        else:
            train_msg = f"Epoch[{epoch+1}/{nb_epochs}] Time: {epoch_duration:.2f}s"
        
        # Validation
        val_msg = ""
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
            
            logs['val_loss'].append(val_metrics['val_loss'])
            logs['val_omnipresent'].append(val_metrics['val_omnipresent'])
            
            val_msg = (f" | Val - Loss: {val_metrics['val_loss']:.4f}, "
                    f"R2: {val_metrics['val_r2']:.4f}, "
                    f"Omni: {val_metrics['val_omnipresent']:.2f}%")
            
            # Early stopping check
            if early_stopper is not None:
                is_best = early_stopper(val_metrics['val_loss'], epoch)
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                    val_msg += " *best*"
                    # Save best model
                    if checkpoint_dir:
                        best_path = os.path.join(checkpoint_dir, f'best_sae_{sae_index}.pt')
                        torch.save(model.state_dict(), best_path)
                else:
                    val_msg += f" (no improv. {early_stopper.counter}/{early_stopper.patience})"
        
        print(train_msg + val_msg)
        
        # Check for early stopping
        if early_stopper is not None and early_stopper.early_stop:
            print(f"\n  [Early Stopping] No improvement for {early_stopper.patience} epochs. "
                  f"Best val_loss: {early_stopper.best_score:.4f} at epoch {early_stopper.best_epoch + 1}")
            
            # Load best model if available
            if checkpoint_dir:
                best_path = os.path.join(checkpoint_dir, f'best_sae_{sae_index}.pt')
                if os.path.exists(best_path):
                    print(f"  [Early Stopping] Loading best model from epoch {early_stopper.best_epoch + 1}")
                    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            break

        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % checkpoint_every_n_epochs == 0:
            early_stopping_state = early_stopper.get_state() if early_stopper else None
            save_checkpoint(checkpoint_dir, sae_index, epoch, global_step, 
                          model, optimizer, scheduler, logs, best_loss, early_stopping_state)

    # Final checkpoint
    if checkpoint_dir:
        early_stopping_state = early_stopper.get_state() if early_stopper else None
        final_epoch = epoch if (early_stopper and early_stopper.early_stop) else nb_epochs - 1
        ckpt_path = save_checkpoint(checkpoint_dir, sae_index, final_epoch, global_step,
                       model, optimizer, scheduler, logs, best_loss, early_stopping_state)
        # Mark training as complete (normal finish or early stop)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        ckpt['training_complete'] = True
        torch.save(ckpt, ckpt_path)

    # Summary
    if val_loader is not None:
        print(f"\n  Training Summary for {sae_prefix}:")
        print(f"    Final train loss: {logs['avg_loss'][-1]:.4f}" if logs['avg_loss'] else "")
        print(f"    Best val loss: {best_val_loss:.4f}")
        if early_stopper:
            print(f"    Best epoch: {early_stopper.best_epoch + 1}")
            print(f"    Stopped at epoch: {epoch + 1}")

    return logs