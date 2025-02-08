from transformers import GPT2Tokenizer
from torch import nn
import torch
class Embedding_Layer(nn.Module) :
    def __init__(self,vocab_size,dim_embedding,context_length):
        super().__init__()
        self.Layer = nn.Embedding(vocab_size,dim_embedding)
        self.Pos_Layer = nn.Embedding(context_length,dim_embedding)
    def forward(self,x):
        word_embedding = self.Layer(x)
        pos_embedding =  self.Pos_Layer(torch.arange(len(x)))
        return word_embedding + pos_embedding
Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = " ana almoudamir"
tokens = Tokenizer.encode(text)
NN = Embedding_Layer(len(Tokenizer.get_vocab()),6,len(tokens))
print(NN(torch.tensor(tokens)))

