import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        """ Processes input text into token IDs """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        """ Converts token IDs back into text """
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

#           Tokenization 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Converting Tokens into Token IDs
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

# Creating Vocabulary dictionary
vocab = {token:integer for integer, token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print("\n Token IDs: ", ids)

decoded_ids = tokenizer.decode(ids) 
print("\n Decoded IDs: ", decoded_ids)

text = "Hello, do you like tea?"
print(tokenizer.encode(text)) # KeyError: 'Hello', "Hello" is not in the Vocabulary