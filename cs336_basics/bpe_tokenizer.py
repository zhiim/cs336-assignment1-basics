import pickle
from collections import defaultdict
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
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        # 使用dict存储merges以及优先级，以高速查找
        self.merges = {merge_pair: i for i, merge_pair in enumerate(merges)}

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
        pre_tokens_pos = defaultdict(list)
        for idx, token in enumerate(token_iter):
            token_str = token.group()
            pre_tokens.append([bytes([i]) for i in token_str.encode("utf-8")])
            # 可能存在重复pre token记录他们出现的位置
            # 遍历的时候直接遍历pre_token_pos，可以减小遍历数量
            pre_tokens_pos[token_str].append(idx)

        for pre_token in pre_tokens_pos.keys():
            cur_token = [bytes([i]) for i in pre_token.encode("utf-8")]
            while True:
                mergeable = []
                for idx in range(len(cur_token) - 1):
                    if (cur_token[idx], cur_token[idx + 1]) in self.merges:
                        mergeable.append((cur_token[idx], cur_token[idx + 1]))
                # 如果当前pre token已经没法合并，直接退出循环
                if len(mergeable) == 0:
                    break
                # 找出优先级最高的merge
                best_merge = min(mergeable, key=lambda x: self.merges[x])
                # 开始merge当前token
                new_pre_token = []
                idx = 0
                while idx < len(cur_token):
                    if idx < len(cur_token) - 1 and best_merge == (
                        cur_token[idx],
                        cur_token[idx + 1],
                    ):
                        new_pre_token.append(best_merge[0] + best_merge[1])
                        idx += 2
                    else:
                        new_pre_token.append(cur_token[idx])
                        idx += 1
                cur_token = new_pre_token
            # 当前pre token已经更新完毕，将它更新到原文本中
            for pos in pre_tokens_pos[pre_token]:
                pre_tokens[pos] = cur_token

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
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        text = b""
        for token_id in ids:
            text += self.vocab[token_id]

        return text.decode("utf-8", errors="replace")
