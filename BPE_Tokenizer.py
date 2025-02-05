import re
import time

#BPE_TOkenizer class
class BPE_tokenize:
    def __init__(self,vocab):
        keys = list(vocab.keys())[:256]
        self.vocab = {keys[i]:i for i in range(len(keys))} # getting first 256 Char
        self.merged = {}
        self.max_id = len(keys)
        self.begining = 1
   
    def split_to_tokens(self,text):
        Words = text.split()
        tokens = []
        for word in Words:
            #check if word contain any of merged pair
            for c in word:
                tokens.append(c)
        return tokens
    def Train(self,text):
        # starting with splitting the tokens to individual char
        text = " ".join([word + "$" for word in text.split()])
        tokens = []
         # spliiting the text to individual bytes
        tokens = self.split_to_tokens(text)
        print(f"tokens----------------------------- {tokens}")
         #getting  the pair and it number of occurence
        for _ in range(10):
            Pair, _ = self.get_Pairs(tokens,text)
            if Pair ==None :
                print("!!!!!!!!!!!!!!!! no Pairs Found STOPP")
                print("------------------------------------------------------")
                print("------------------------------------------------------")
                print(f"current vocab f{self.vocab}")
                print(f"merged words f{self.merged}")
                return 1
        #adding new pair two the merged list
            self.merged[f"<{self.max_id}>"] = Pair
            self.vocab[Pair[0]+Pair[1]] = self.max_id
            tokens = self.merging_tokens(tokens,Pair)
            #sub current teext with new Pair
            text = re.sub(re.escape(Pair[0]+Pair[1]),f"<{self.max_id}>",text)
            self.max_id+=1
            print(f"new tokens--------------------------------- {tokens}")
            print(f" new text {text}")
            time.sleep(1)
    def merging_tokens(self,tokens,Pair):
        #merging tokens if there is pairs existing
        merged_token = []
        i =0 
        while i < len(tokens)-1:
            if tokens[i] == Pair[0] and tokens[i+1] == Pair[1]:
                merged_token.append(f'<{self.max_id}>')
                i+=2
            else:
                merged_token.append(tokens[i])
                i+=1

        return merged_token


    def get_Pairs(self,tokens,text):
        #find most occurent pair in sequence of tokens
        Count = {}
        for i in range(len(tokens) -1):
             Pair = (tokens[i],tokens[i+1])
             occurence = max(0,len(re.findall(re.escape(Pair[0]+Pair[1]),text)))
             if occurence < 1 :
                 continue
             Count[Pair] = occurence
        print(Count)
        if Count == {} :
            return None,0
        return sorted(Count.items(),key= lambda x: -x[1])[0]
        

tokenizer = BPE_tokenize({})
status =tokenizer.Train("the cat in the hat")
