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

        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        # add special tokens
        if special_tokens is not None:
            for token in special_tokens:
                token_encode = token.encode("utf-8")
                if token_encode not in self.vocab_reverse:
                    self.vocab[len(self.vocab)] = token_encode
                    self.vocab_reverse[token_encode] = len(self.vocab_reverse)

        self.special_tokens = special_tokens

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
        token_ids = []

        if self.special_tokens is not None:
            # 先对special tokens根据长短排序，防止某些较长的special token被前面
            # 的较短special token切分
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens = [re.escape(token) for token in special_tokens]
            split_pattern = "|".join(special_tokens)

            docs = re.split(f"({split_pattern})", text)

            for doc in docs:
                token_ids.extend(self._encode_doc(doc))
        else:
            token_ids = self._encode_doc(text)

        return token_ids

    def _encode_doc(self, doc) -> list[int]:
        # directly encode special tokens
        if self.special_tokens is not None and doc in self.special_tokens:
            return [self.vocab_reverse[doc.encode("utf-8")]]

        # pre-tokenization first
        token_iter = re.finditer(self.pat, doc)
        pre_tokens = []
        for token in token_iter:
            pre_tokens.append(
                [bytes([i]) for i in token.group().encode("utf-8")]
            )

        for pair in self.merges:
            new_pre_tokens = []
            for token in pre_tokens:
                new_token = []
                idx = 0
                while idx < len(token):
                    if idx <= len(token) - 2 and pair == (
                        token[idx],
                        token[idx + 1],
                    ):
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
            token_ids =  self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        text = b""
        for token_id in ids:
            text += self.vocab[token_id]

        return text.decode("utf-8", errors="replace")
