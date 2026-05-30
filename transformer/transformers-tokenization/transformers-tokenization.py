import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.id_to_word[0] = "<PAD>"
        self.word_to_id["<PAD>"] = 0
        self.id_to_word[1] = "<UNK>"
        self.word_to_id["<UNK>"] = 1
        self.id_to_word[2] = "<BOS>"
        self.word_to_id["<BOS>"] = 2
        self.id_to_word[3] = "<EOS>"
        self.word_to_id["<EOS>"] = 3
        self.vocab_size = 4
        corpus = set()
        idx = 4
        for text in texts:
            for word in text.lower().split():
                corpus.add(word)

        for word in sorted(corpus):
            self.id_to_word[idx] = word
            self.word_to_id[word] = idx
            idx += 1
            self.vocab_size += 1
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        encoded_ids = []
        for word in text.lower().split():
            if word in self.word_to_id:
                encoded_ids.append(self.word_to_id[word])
            else:
                encoded_ids.append(1)

        return encoded_ids
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        decoded_ids = []
        for id in ids:
            if id in self.id_to_word:
                decoded_ids.append(self.id_to_word[id])
            else:
                decoded_ids.append("<UNK>")

        return " ".join(decoded_ids)
        pass
