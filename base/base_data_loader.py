"""Base DataLoader class."""
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

class BaseDataLoader(DataLoader):
    """Base class for all data loaders with train/val split support."""
    
    def __init__(self, dataset, batch_size, shuffle, validation_split=0.0, num_workers=0):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.n_samples = len(dataset)
        
        self.sampler, self.valid_sampler = self._split_sampler(validation_split)
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle if self.sampler is None else False,
            'num_workers': num_workers,
            'pin_memory': True
        }
        
        if self.sampler:
            super().__init__(sampler=self.sampler, **self.init_kwargs)
        else:
            super().__init__(**self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx = np.arange(self.n_samples)
        np.random.seed(42)
        np.random.shuffle(idx)

        split_idx = int(self.n_samples * split)
        train_idx, valid_idx = idx[split_idx:], idx[:split_idx]
        
        self.n_samples = len(train_idx)
        return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

    def get_valid_loader(self):
        if self.valid_sampler is None:
            return None
        return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
