import re
import torch
import random

class Data:
    def __init__(self, path, expand_factor=1, special_char="<.>"):
        """
        - Takes a .txt file and tokenises the complete text
        - Has two dictionaries, [wtoi] to convert words to integer and [itow] to do the vice versa
        - Special character is to end and begin the sentences, it is given the fix integer 0 
        """
        text_lines = open(path).read().lower().splitlines()
        self.line_tokens = list() 
        text_tokens = list() 

        # making tokens of each line
        for line in text_lines:
            words = re.findall(r'\w+|[^\w\s]|\n', line)
            if words:
                self.line_tokens.append(words)
                for word in words:
                    text_tokens.append(word)

        vocab = sorted(list(set(text_tokens)))
        self.special_char = special_char
        
        self.wtoi = {w: i+1 for i, w in enumerate(vocab)}
        self.wtoi[self.special_char] = 0
        self.itow = {i:w for w, i in self.wtoi.items()}

        self.line_tokens = self.line_tokens * expand_factor
        self.vocab_size = len(self.wtoi)


    def build_dataset(self, context_size=4, line_tokens=None):
        """
        - Forms the data into X and Y, X being the context and Y being the output.
        - Uses a dictionary (from __init__) to convert the words to integers.
        """
        X, Y = [], []
        if not line_tokens:
            line_tokens = self.line_tokens
        for line in line_tokens:
            context_window = [0] * context_size
            for word in line + [self.special_char]:
                ix = self.wtoi[word]
                X.append(context_window)
                Y.append(ix)
                context_window = context_window[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y

    def build_dataset_w_split(self, context_size=4, train_split=0.95, val_split=0.025, test_split=0.025, random_seed=42) -> None:
        """
        - This function will make splits of the dataset, based on given by user or deault.
        - It also take context_size, that is X's no. of features.
        """
        assert train_split + val_split + test_split == 1 and (train_split >=0.0 and val_split >= 0.0 and test_split >= 0.0), "incorrect split values, make sure the account ofr all the data exactly"
        random.seed(random_seed)
        random.shuffle(self.line_tokens)
        total_length = len(self.line_tokens)
        n1, n2 = int(train_split*total_length), int((train_split+val_split)*total_length)
        self.Xtr, self.Ytr = self.build_dataset(line_tokens=self.line_tokens[:n1], context_size=context_size)
        self.Xdev, self.Ydev = self.build_dataset(line_tokens=self.line_tokens[n1:n2], context_size=context_size)
        self.Xte, self.Yte = self.build_dataset(line_tokens=self.line_tokens[n2:], context_size=context_size)


