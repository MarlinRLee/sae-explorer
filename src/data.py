import os
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class ShardedDataset(IterableDataset):
    def __init__(self, shard_dir, batch_size_per_worker, shuffle_shards=True, shuffle_in_shard=True, subset_fraction=1.0):
        super().__init__()
        self.shard_dir = shard_dir
        self.batch_size_per_worker = batch_size_per_worker 
        self.shuffle_shards = shuffle_shards
        self.shuffle_in_shard = shuffle_in_shard
        
        self.shard_files = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.pt')))
        
        if subset_fraction < 1.0:
            rng = np.random.RandomState(42)
            rng.shuffle(self.shard_files)
            num_to_keep = max(1, int(len(self.shard_files) * subset_fraction))
            self.shard_files = self.shard_files[:num_to_keep]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            shard_indices = list(range(len(self.shard_files)))
        else:
            shard_indices = list(range(worker_info.id, len(self.shard_files), worker_info.num_workers))

        if self.shuffle_shards:
            # Use OS-seeded RNG so each worker gets a unique shuffle
            rng = np.random.default_rng()
            rng.shuffle(shard_indices)

        for shard_idx in shard_indices:
            try:
                # Load to CPU, keep in fp16 if saved that way
                data = torch.load(self.shard_files[shard_idx], map_location='cpu', weights_only=True)
                
                if self.shuffle_in_shard:
                    idx = torch.randperm(data.shape[0])
                    data = data[idx]

                total_samples = data.shape[0]
                # Yield chunks to reduce Python overhead
                for i in range(0, total_samples, self.batch_size_per_worker):
                    yield data[i : i + self.batch_size_per_worker]
                    
            except Exception as e:
                print(f"Error loading {self.shard_files[shard_idx]}: {e}")

class DeviceDataLoader:
    """Wraps a dataloader to move batches to device."""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            batch = batch.to(self.device, non_blocking=True).float()
            yield batch

    def __len__(self):
        return len(self.dataloader)

# --- 4. The Setup Function ---
def create_dataloader(shard_dir, total_batch_size, num_workers=9, prefetch_factor=6, 
                      subset_fraction=1.0, shuffle=True):
    """
    Create a dataloader for sharded data.
    
    Parameters
    ----------
    shard_dir : str
        Directory containing shard_*.pt files.
    total_batch_size : int
        Total batch size across all workers.
    num_workers : int
        Number of data loading workers.
    prefetch_factor : int
        Number of batches to prefetch per worker.
    subset_fraction : float
        Fraction of shards to use (for faster iteration during development).
    shuffle : bool
        Whether to shuffle shards and within shards.
    
    Returns
    -------
    DataLoader
        PyTorch DataLoader instance.
    """
    batch_per_worker = total_batch_size // num_workers
    if total_batch_size % num_workers != 0:
        effective = batch_per_worker * num_workers
        print(f"[Warning] total_batch_size ({total_batch_size}) not divisible by "
              f"num_workers ({num_workers}). Effective batch size: {effective}")

    dataset = ShardedDataset(
        shard_dir, 
        batch_size_per_worker=batch_per_worker,
        shuffle_shards=shuffle,
        shuffle_in_shard=shuffle,
        subset_fraction=subset_fraction
    )
    
    def fast_collate(batch_list):
        return torch.cat(batch_list, dim=0)

    return DataLoader(
        dataset,
        batch_size=max(1, num_workers),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        collate_fn=fast_collate,
        drop_last=True
    )


def create_val_dataloader(shard_dir, total_batch_size, num_workers=4, prefetch_factor=2,
                          subset_fraction=1.0):
    """
    Create a validation dataloader.
    
    Validation uses less aggressive settings since we only iterate once per epoch.
    No shuffling for reproducibility.
    
    Parameters
    ----------
    shard_dir : str
        Directory containing validation shard_*.pt files.
    total_batch_size : int
        Total batch size across all workers.
    num_workers : int
        Number of data loading workers (fewer than training).
    prefetch_factor : int
        Number of batches to prefetch per worker.
    subset_fraction : float
        Fraction of shards to use. Use 1.0 to use all validation data,
        or smaller value for faster validation during development.
    
    Returns
    -------
    DataLoader
        PyTorch DataLoader instance for validation.
    """
    batch_per_worker = total_batch_size // max(1, num_workers)
    
    dataset = ShardedDataset(
        shard_dir, 
        batch_size_per_worker=batch_per_worker,
        shuffle_shards=False,  # No shuffling for validation
        shuffle_in_shard=False,  # Deterministic order
        subset_fraction=subset_fraction
    )
    
    def fast_collate(batch_list):
        return torch.cat(batch_list, dim=0)

    return DataLoader(
        dataset,
        batch_size=max(1, num_workers),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None, 
        persistent_workers=False,
        collate_fn=fast_collate,
        drop_last=False  # Use all validation samples
    )