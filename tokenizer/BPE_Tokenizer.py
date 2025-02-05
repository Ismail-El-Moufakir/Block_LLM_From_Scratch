import re

class BPE_tokenize:
    def __init__(self):
        # Initialisation du vocabulaire avec les 256 premiers caractères ASCII
        self.vocab = {chr(i): i for i in range(256)}
        self.id_to_token = {i: chr(i) for i in range(256)}
        self.merged = {}  # Stocke les paires fusionnées
        self.max_id = len(self.vocab)  # ID maximal actuel

    def split_to_tokens(self, text):
        """Découpe le texte en caractères individuels avec marqueur de fin `$`."""
        return list(text)

    def get_Pairs(self, tokens):
        """Trouve la paire la plus fréquente dans la liste des tokens."""
        Count = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            Count[pair] = Count.get(pair, 0) + 1
        
        if not Count:
            return None, 0
        return max(Count.items(), key=lambda x: x[1])  # Retourne la paire la plus fréquente

    def merging_tokens(self, tokens, pair):
        """Fusionne toutes les occurrences de `pair` en un seul token."""
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged_tokens.append(pair[0] + pair[1])  # Fusion des deux caractères
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def Train(self, text, num_merges=10):
        """Entraîne le tokeniseur BPE sur un texte donné."""
        text = " ".join([word + "$" for word in text.split()])  # Ajoute des marqueurs de fin de mot
        tokens = self.split_to_tokens(text)

        for _ in range(num_merges):
            pair, occurrence = self.get_Pairs(tokens)
            if pair is None:
                break  # Arrête si plus aucune paire ne peut être fusionnée

            # Ajout de la nouvelle paire au vocabulaire et fusion des tokens
            new_token = pair[0] + pair[1]
            self.merged[pair] = occurrence
            self.vocab[new_token] = self.max_id
            self.id_to_token[self.max_id] = new_token
            self.max_id += 1
            tokens = self.merging_tokens(tokens, pair)

        print(f"Vocab final : {self.vocab}")
        print(f"Paires fusionnées : {self.merged}")

    def encode(self, text):
        """Encode un texte en une séquence d'IDs."""
        text = " ".join([word + "$" for word in text.split()])
        tokens = self.split_to_tokens(text)

        for pair in sorted(self.merged.keys(), key=lambda p: self.vocab.get(p[0] + p[1], float('inf'))):
            tokens = self.merging_tokens(tokens, pair)

        token_ids = [self.vocab[token] for token in tokens]  # Convertit les tokens en IDs
        return token_ids

    def decode(self, token_ids):
        """Reconstitue le texte original à partir des IDs."""
        tokens = [self.id_to_token[id] for id in token_ids]
        text = "".join(tokens).replace("$", " ").strip()
        return text


# Test du tokeniseur BPE
tokenizer = BPE_tokenize()
text = "Neural networks are a subset of machine learning, inspired by the structure of the human brain."

# Entraînement du tokeniseur
tokenizer.Train(text, num_merges=10)

# Encodage d'une nouvelle phrase en IDs
tokens_ids = tokenizer.encode("ana almoudamir ana almoudamir")
print("\nTokens (IDs):", tokens_ids)

# Décodage depuis les IDs
decoded_text = tokenizer.decode(tokens_ids)
print("\nTexte décodé:", decoded_text)
