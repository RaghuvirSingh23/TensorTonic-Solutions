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
        self.word_to_id = {}
        self.id_to_word = {}

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]

        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        corpus = set()

        for text in texts:
            for word in text.lower().split():
                corpus.add(word)

        idx = len(special_tokens)

        for word in sorted(corpus):
            if word not in self.word_to_id:
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
                idx += 1

        self.vocab_size = idx

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoded_ids = []

        for word in text.lower().split():
            encoded_ids.append(
                self.word_to_id.get(word, self.word_to_id[self.unk_token])
            )

        return encoded_ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded_words = []

        for token_id in ids:
            decoded_words.append(
                self.id_to_word.get(token_id, self.unk_token)
            )

        return " ".join(decoded_words)