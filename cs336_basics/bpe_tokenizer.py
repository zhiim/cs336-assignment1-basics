import pickle
from typing import Iterable

import regex as re


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and
        (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges

        # add special tokens
        for token in special_tokens:
            token_encode = token.encode("utf-8")
            if token_encode not in self.vocab:
                self.vocab[len(self.vocab)] = token_encode

        self.special_tokens = special_tokens

        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # noqa

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized
        vocabulary and list of merges (in the same format that your BPE training
        code output) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""
        special_tokens = [re.escape(token) for token in self.special_tokens]
        split_pattern = "|".join(special_tokens)

        docs = re.split(f"({split_pattern})", text)

        token_ids = []
        for doc in docs:
            token_ids.extend(self._encode_doc(doc))

        return token_ids

    def _encode_doc(self, doc) -> list[int]:
        # directly encode special tokens
        if doc in self.special_tokens:
            return [self.vocab_reverse[doc.encode("utf-8")]]

        # pre-tokenization first
        token_iter = re.finditer(self.pat, doc)
        pre_tokens = []
        for token in token_iter:
            pre_tokens.append([bytes([i]) for i in token.encode("utf-8")])

        for pair in self.merges:
            new_pre_tokens = []
            for token in pre_tokens:
                new_token = []
                idx = 0
                while idx <= len(token) - 2:
                    if pair == (token[idx], token[idx + 1]):
                        new_token.append(token[idx] + token[idx + 1])
                        idx += 2
                    else:
                        new_token.append(token[idx])
                        idx += 1
                new_pre_tokens.append(new_token)
            pre_tokens = new_pre_tokens

        token_ids = []
        for token in pre_tokens:
            token_ids.extend([self.vocab_reverse[item] for item in token])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Given an iterable from_files strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs. This is required for
        memory-eﬀicient tokenization of large files that we cannot directly load
        into memory."""
        for text in iterable:
            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        text = b""
        for token_id in ids:
            text += self.vocab[token_id]

        return text.decode("utif-0", errors="replace")
