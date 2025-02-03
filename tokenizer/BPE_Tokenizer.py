import re
from transformers import GPT2Tokenizer

class BPE_Tokenizer:
    def __init__(self, vocab):
        self.vocab = {token: i for i, token in enumerate(list(vocab.keys())[:256])}  # Limit to first 256 tokens
        self.tokens_freq = {}
        self.max_id = max(self.vocab.values())  # Start from highest existing token ID
        self.merged_tokens = {}

    def init_Tokenizer(self, text):
        """
        Initialize tokenizer by splitting text into words and subwords, then computing initial token frequencies.
        """
        self.text = " ".join([word + "</w>" for word in text.split()])  # Preserve word boundaries
        self.tokens_freq = {}

        for token in self.text.split():
            if token in self.tokens_freq:
                self.tokens_freq[token] += 1
            else:
                self.tokens_freq[token] = 1  # Initialize frequency count

    def found_Pair(self):
        """
        Finds the most frequently occurring adjacent token pair in the text without using Counter.
        """
        token_list = self.text.split()
        pair_freq = {}
        most_frequent_pair = None
        max_count = 0

        # Iterate over all adjacent token pairs
        for i in range(len(token_list) - 1):
            pair = (token_list[i], token_list[i + 1])
            pair_str = " ".join(pair)

            # Count occurrences manually
            if pair_str in pair_freq:
                pair_freq[pair_str] += 1
            else:
                pair_freq[pair_str] = 1

            # Keep track of the most frequent pair
            if pair_freq[pair_str] > max_count:
                most_frequent_pair = pair
                max_count = pair_freq[pair_str]

        if not most_frequent_pair:
            return "", 0, []

        return " ".join(most_frequent_pair), max_count, most_frequent_pair

    def Train(self, num_merges=10):
        """
        Trains the tokenizer by merging the most frequent token pairs.
        """
        for iteration in range(num_merges):
            new_token, nbre_occurence, old_tokens = self.found_Pair()
            if not new_token:
                break  # Stop if no more frequent pairs exist

            print(f"Iteration {iteration + 1}: Merging {old_tokens} -> {new_token}")

            # Update vocabulary
            self.max_id += 1
            self.vocab[new_token] = self.max_id
            self.merged_tokens[new_token] = f"<{self.max_id}>"

            # Replace occurrences in text
            old_pair_str = " ".join(old_tokens)
            self.text = self.text.replace(old_pair_str, f"<{self.max_id}>")

            # Update token frequencies manually
            self.tokens_freq[f"<{self.max_id}>"] = nbre_occurence
            for token in old_tokens:
                if token in self.tokens_freq:
                    self.tokens_freq[token] -= nbre_occurence

            print(f"Updated Text: {self.text}\n")
        
        print("Final Vocabulary:", self.vocab)

# Example Usage
GPT2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab = GPT2_tokenizer.get_vocab()
tokenizer = BPE_Tokenizer(vocab)

text = "ana almoudamir ana almoudamir"
tokenizer.init_Tokenizer(text)
tokenizer.Train(num_merges=5)
