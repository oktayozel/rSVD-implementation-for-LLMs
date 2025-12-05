"""
Prepare the WikiText-103 dataset for token-level language modeling.
WikiText-103 is ~100M tokens, good for medium-sized models.
"""
import os
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset

# Load WikiText-103 from HuggingFace
print("Loading WikiText-103 dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Get train and validation splits
print("Concatenating train data...")
train_data = '\n'.join(dataset['train']['text'])
print("Concatenating validation data...")
val_data = '\n'.join(dataset['validation']['text'])

print(f"Train dataset length: {len(train_data):,} characters")
print(f"Val dataset length: {len(val_data):,} characters")

# Use GPT-2 BPE tokenizer
print("Initializing tokenizer...")
enc = tiktoken.get_encoding("gpt2")

# Encode datasets
print("Encoding train data...")
train_ids = enc.encode_ordinary(train_data)
print("Encoding validation data...")
val_ids = enc.encode_ordinary(val_data)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Save to binary files
print("Saving train.bin...")
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))

print("Saving val.bin...")
val_ids = np.array(val_ids, dtype=np.uint16)
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save meta information
meta = {
    'vocab_size': 50304,  # GPT-2 vocab size padded
    'tokenizer': 'gpt2_bpe',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done!")
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens: {len(val_ids):,}")