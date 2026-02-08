"""RefCOCO Dataset with LMDB backend."""
import os
import io
import pickle
import random
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer

def cache_lmdb_keys(lmdb_path, output_path=None):
    """Cache LMDB keys to pickle file for faster access."""
    if output_path is None:
        output_path = str(lmdb_path) + '.keys.pkl'

    if os.path.exists(output_path):
        print(f'Loading cached keys from {output_path}...')
        with open(output_path, 'rb') as f:
            keys = pickle.load(f)
        print(f'Loaded {len(keys)} keys')
        return keys

    print(f'Extracting keys from {lmdb_path}...')
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, 
                    readahead=False, meminit=False, subdir=False)
    keys = []
    with env.begin(write=False) as txn:
        for key, value in txn.cursor():
            data = pickle.loads(value)
            if isinstance(data, dict):
                keys.append(key)
    env.close()

    with open(output_path, 'wb') as f:
        pickle.dump(keys, f)
    print(f'Cached {len(keys)} keys')
    return keys


class RefCOCODataset(Dataset):
    """RefCOCO Dataset with LMDB backend."""
    
    def __init__(self, lmdb_path, config, transform=None):
        self.lmdb_path = lmdb_path
        self.config = config
        self.transform = transform
        self.env = None
        
        self.keys = cache_lmdb_keys(lmdb_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.text_encoder)

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False,
                                 readahead=False, meminit=False, subdir=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_env()
        
        with self.env.begin(write=False) as txn:
            data = pickle.loads(txn.get(self.keys[idx]))

        # Load image
        img = Image.open(io.BytesIO(data['img'])).convert('RGB')
        orig_w, orig_h = img.size

        # Get text
        sents = data['sents']
        text = random.choice(sents) if isinstance(sents, list) else sents

        # Compute bbox from mask
        mask = np.array(Image.open(io.BytesIO(data['mask'])))
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        else:
            x1, y1, x2, y2 = 0, 0, orig_w, orig_h

        bbox = torch.tensor([x1/orig_w, y1/orig_h, x2/orig_w, y2/orig_h], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        tokens = self.tokenizer(text, max_length=self.config.max_text_len,
                                padding='max_length', truncation=True, return_tensors='pt')

        return {
            'image': img,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'bbox': bbox,
            'text': text
        }
