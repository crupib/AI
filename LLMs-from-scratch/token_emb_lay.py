import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

vocab_size = 50257
output_dim = 256

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]  # Fixed typo here
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# Handle file reading with error checking
file_path = "the-verdict.txt"
if not os.path.exists(file_path):
    print(f"Note: File '{file_path}' not found. Using sample text instead.")
    # Create sample text for testing
    raw_text = "This is a sample text for testing the dataloader functionality. " * 100
else:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

# Create dataloader and test
max_length=4
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

context_length = max_length
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
token_embedded_layer = nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedded_layer(inputs)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
