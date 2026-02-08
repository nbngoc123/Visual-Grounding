"""DataLoader for RefCOCO."""
from pathlib import Path
from torchvision import transforms as T

from base import BaseDataLoader
from .datasets import RefCOCODataset


def get_transform(img_size):
    """Image transforms (no crop to preserve bbox alignment)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class RefCOCODataLoader(BaseDataLoader):
    """RefCOCO DataLoader."""
    
    def __init__(self, data_dir, config, split='train', batch_size=None, 
                 shuffle=True, validation_split=0.0, num_workers=0):
        
        lmdb_path = Path(data_dir) / 'lmdb' / 'refcoco' / f'{split}.lmdb'
        transform = get_transform(config.img_size)
        
        self.dataset = RefCOCODataset(lmdb_path, config, transform)
        
        super().__init__(
            self.dataset,
            batch_size=batch_size or config.batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers
        )
