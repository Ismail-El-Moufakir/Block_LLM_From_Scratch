
import re

class Simple_Tokenizer:
    def __init__(self,token_id):
        self.token_id = token_id
        self.id_token = {i:token for token,i in enumerate(self.token_id)}
    def init_vocab(self,Tokens):
        '''
        initialize vocabulary set
        '''
        if self.token_id == {}:
            #if vocab is empty choose tokens as vocab
            Tokens_Set = list(set(sorted(Tokens)))
            self.token_id =  {Tokens_Set[i]:i for i in range(len(Tokens_Set))}
            self.id_token =  {i:Tokens_Set[i] for i in range(len(Tokens_Set))}
            #adding unkown case to the vocab
        self.token_id["unk"] = max(self.token_id.values()) +1
        self.id_token[max(self.token_id.values()) +1]  = "unk"

    def encode(self,text):
        '''
        encode the sequence of words into a list of token ids
        '''
        # first we need to split the sequence into tokens
        Tokens = re.split("[-/&()|%$\s]",text)
        #removing every whitespace
        for t in Tokens:
            if t == "":
                Tokens.remove(t)
        # converting the tokens to ids
        self.init_vocab(Tokens) 
        Tokens_ids = []
        for token in Tokens:
            Tokens_ids.append(self.token_id[token])
        return Tokens_ids

    def decode(self,tokens_ids):
        '''
        decode a list of token ids  into text
        '''
        return " ".join([self.id_token[i] for i in tokens_ids])

tokenizer = Simple_Tokenizer({})
tokens = tokenizer.encode("Dr. Smith's AI-powered system detected 3,456 anomalies in less than 2.5 seconds! Can you believe it? Meanwhile, María said: «C'est incroyable!»")
print(f"text encoded:m {tokens}")
print("-------------------------------------------")
print(f"text decoded {tokenizer.decode(tokens_ids=tokens)}")