"""
Prepare the WikiText-2 dataset for character-level or token-level language modeling.
"""
import os
import pickle
import requests
import numpy as np
import tiktoken
from datasets import load_dataset

# Load WikiText-2 from HuggingFace
print("Loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Get train and validation splits
train_data = '\n'.join(dataset['train']['text'])
val_data = '\n'.join(dataset['validation']['text'])

print(f"Train dataset length: {len(train_data):,} characters")
print(f"Val dataset length: {len(val_data):,} characters")

# Option 1: Character-level encoding (like shakespeare_char)
def prepare_char_level():
    # Get unique characters
    chars = sorted(list(set(train_data + val_data)))
    vocab_size = len(chars)
    print(f"Vocab size (characters): {vocab_size}")
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # Encode datasets
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")
    
    # Save to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    # Save meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

# Option 2: BPE token-level encoding (like openwebtext)
def prepare_token_level():
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode datasets
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")
    
    # Save to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    
    # Save meta information
    meta = {
        'vocab_size': 50304,  # GPT-2 vocab size padded
        'tokenizer': 'gpt2_bpe',
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    # Choose encoding type
    # prepare_char_level()  # For character-level modeling
    prepare_token_level()   # For token-level modeling (recommended for medium model)