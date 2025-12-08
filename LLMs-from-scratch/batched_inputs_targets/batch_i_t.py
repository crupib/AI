import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import os


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
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print("Input shape:", first_batch[0].shape)
print("Target shape:", first_batch[1].shape)
print("\nFirst batch - Input:", first_batch[0])
print("First batch - Target:", first_batch[1])
second_batch = next(data_iter)
print("\nSecond batch - Input:", second_batch[0])
print("Second batch - Target:", second_batch[1])