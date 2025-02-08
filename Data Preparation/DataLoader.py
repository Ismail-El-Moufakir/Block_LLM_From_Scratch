from BPE_tokenizer import BPE_tokenizer
from torch.utils.data import Dataset, DataLoader
import torch

class Text_DataSet(Dataset):
    def __init__(self, raw_text, tokenizer, stride, max_length):
        self.tokens = tokenizer.encode(raw_text)  # Ensure tokenizer has an `encode()` method
        self.input = []
        self.target = []

        # Iterate correctly with stride
        for i in range(0, len(self.tokens) - max_length, stride):
            self.input.append(torch.tensor(self.tokens[i : i + max_length], dtype=torch.long))
            self.target.append(torch.tensor(self.tokens[i + 1 : i + max_length + 1], dtype=torch.long))

        self.input = torch.stack(self.input)
        self.target = torch.stack(self.target)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

text = "ana almoudamir ayouha nass"
BPE_tokenize = BPE_tokenizer()
BPE_tokenize.Train(text)  
Ds = Text_DataSet(text, BPE_tokenize, stride=4, max_length=4)
Loader = DataLoader(Ds, batch_size=2, shuffle=True)

# Check batch output
for batch in Loader:
    inputs, targets = batch
    print("Input:", inputs)
    print("Target:", targets)
    break
