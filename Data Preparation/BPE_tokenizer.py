import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def train(self, corpus):
        token_freqs = Counter()
        for word in corpus:
            tokens = list(word) + ['</w>']
            token_freqs[tuple(tokens)] += 1
        
        while len(self.vocab) < self.vocab_size:
            pairs = defaultdict(int)
            for token_seq, freq in token_freqs.items():
                for i in range(len(token_seq) - 1):
                    pairs[(token_seq[i], token_seq[i+1])] += freq
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = ''.join(best_pair)
            self.merges.append(best_pair)
            new_token_freqs = Counter()
            for token_seq, freq in token_freqs.items():
                new_seq = []
                i = 0
                while i < len(token_seq):
                    if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:
                        new_seq.append(new_token)
                        i += 2
                    else:
                        new_seq.append(token_seq[i])
                        i += 1
                new_token_freqs[tuple(new_seq)] += freq
            token_freqs = new_token_freqs
        
        self.vocab = {token: idx for idx, token in enumerate(token_freqs.keys())}
    
    def encode(self, word):
        tokens = list(word) + ['</w>']
        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            merges = {pair: ''.join(pair) for pair in self.merges}
            min_index = float('inf')
            best_pair = None
            for pair in pairs:
                if pair in merges and self.merges.index(pair) < min_index:
                    min_index = self.merges.index(pair)
                    best_pair = pair
            if best_pair is None:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, tokens):
        return ''.join(tokens).replace('</w>', '')
