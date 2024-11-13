from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = cpu_count() // 2
print(f"Using {num_proc} processes for {cpu_count()} cores")

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("openwebtext")

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids}
    return out

# First process validation set
print("Processing validation set...")
val_tokenized = split_dataset['val'].map(
    process,
    remove_columns=['text'],
    desc="tokenizing validation split",
    num_proc=num_proc,
)

# Save validation set
print("writing val.bin...")
val_tokens = np.concatenate([example['ids'] for example in tqdm(val_tokenized)])
val_tokens = val_tokens.astype(np.uint16)
val_tokens.tofile('val.bin')

# Then process training set in chunks
print("Processing training set...")
train_tokenized = split_dataset['train'].map(
    process,
    remove_columns=['text'],
    desc="tokenizing training split", 
    num_proc=num_proc,
)

# Save training set in chunks
print("writing train.bin...")
chunk_size = 729_754  # should work on 16GB
total_chunks = (len(train_tokenized)-1)//chunk_size + 1

with open('train.bin', 'wb') as f:
    for i in tqdm(range(0, len(train_tokenized), chunk_size), 
                 desc="Processing chunks", 
                 total=total_chunks):
        chunk = train_tokenized.select(range(i, min(i + chunk_size, len(train_tokenized))))
        chunk_tokens = np.concatenate([example['ids'] for example in chunk])
        chunk_tokens = chunk_tokens.astype(np.uint16)
        chunk_tokens.tofile(f)

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')