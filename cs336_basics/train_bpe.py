import logging
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # noqa


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    Chunk boundaries occur at the beginning of a special token.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than
    # desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenization(args):
    file_path, split_pattern, start, end = args
    pre_tokens_count = Counter()

    with open(file_path, "rb") as file:
        file.seek(start)  # read chunk in utf-8 string, not bytes
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

        # 根据special token将每个chunk划分成多个doc段
        splited_doc = re.split(split_pattern, chunk)

        for doc in splited_doc:
            # 在每个doc内进行pre-tokenization
            match_iterator = re.finditer(PAT, doc)
            for token in match_iterator:
                pre_tokens_count[tuple(token.group().encode("utf-8"))] += 1

    logging.debug(f"pre-tokenization in chunk: {start}-{end} finished")
    return pre_tokens_count


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_processes=32,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Training a BPE tokenizer

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary
            size.
        special_tokens: A list of strings to add to the vocabulary. These
            special tokens do not otherwise affect BPE training

    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the
            vocabulary) to bytes (token bytes).
        merges: A list of BPE merges produced from training. Each list item is a
            tuple of bytes (<token1>, <token2>), representing that <token1> was
            merged with <token2>. The merges should be ordered by order of
            creation.
    """
    # -- 1. initialize vocabulary ------------------------------------------
    logging.info("1. initialize vocabulary")

    # all 256 possible byte values
    vocab = {i: bytes([i]) for i in range(256)}

    # and special tokens
    new_ids = len(vocab)
    for token in special_tokens:
        vocab[new_ids] = token.encode("utf-8")
        new_ids += 1

    logging.info("vocabulary initiaed")
    logging.info("-" * 40)

    # -- 2. pre-tokenization -----------------------------------------------
    logging.info("2. pre-tokenization")

    pre_tokens_count = Counter()  # 用于存储pre token和它们对于的计数

    # divide the whole text file into chunks
    first_special_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, first_special_token
        )
    logging.info("chunk boundaries found")

    # read chunk one by one
    special_tokens = [re.escape(token) for token in special_tokens]
    split_pattern = "|".join(special_tokens)

    args_list = [
        (input_path, split_pattern, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool() as pool:
        results = pool.map(pre_tokenization, args_list)
    for result in results:
        pre_tokens_count.update(result)

    logging.info("pre-tokenization finied")
    logging.info("-" * 40)

    # -- 3. BPE merges -----------------------------------------------------
    logging.info("3. BPE merge")

    merges = []

    pairs_count = defaultdict(int)
    pairs_in_pre_token = defaultdict(set)  # 记录当前pair出现在哪些pre token中
    # 对于每个pre token的每两个相邻byte，计算出现次数
    for pre_token, count in pre_tokens_count.items():
        for idx1, idx2 in zip(pre_token, pre_token[1:]):
            pairs_count[(idx1, idx2)] += count
            pairs_in_pre_token[(idx1, idx2)].add(pre_token)
    logging.info("all pairs counting finished")

    # vocab size达到要求就结束
    vocab_size_start = len(vocab)
    for new_id in range(vocab_size_start, vocab_size):
        logging.info(
            f"merge loop {new_id - vocab_size_start + 1}/{vocab_size - vocab_size_start}"  # noqa
        )
        # 找到出现次数最多的pair，并且字典序最大的
        merge_pair, _ = max(
            pairs_count.items(),
            key=lambda items: (
                items[1],
                (vocab[items[0][0]], vocab[items[0][1]]),
            ),
        )

        # 添加到vocab中
        new_bytes = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        vocab[new_id] = new_bytes

        merges.append((vocab[merge_pair[0]], vocab[merge_pair[1]]))

        # 根据新的merge，重新计算pairs_count，并且更新pre token
        logging.info("updating pre-tokens for merging ...")
        for pre_token in list(
            pairs_in_pre_token[merge_pair]
        ):  # 只需要更新包含了merge pair的pre token
            if pre_token not in pre_tokens_count:
                continue

            count = pre_tokens_count[pre_token]

            # 将所有和旧pre token的相关记录全部清除
            pre_tokens_count.pop(pre_token)
            for pair in zip(pre_token, pre_token[1:]):
                pairs_count[pair] -= count
                if pairs_count[pair] == 0:
                    pairs_count.pop(pair)

                if (
                    pair in pairs_in_pre_token
                    and pre_token in pairs_in_pre_token[pair]
                ):
                    pairs_in_pre_token[pair].remove(pre_token)
                    if not pairs_in_pre_token[pair]:
                        pairs_in_pre_token.pop(pair)

            # 根据merge pair更新pre token
            new_pre_token = []
            idx = 0
            while idx <= len(pre_token) - 1:
                # 如果发现了merge pair
                if (
                    idx <= len(pre_token) - 2
                    and (pre_token[idx], pre_token[idx + 1]) == merge_pair
                ):
                    new_pre_token.append(new_id)
                    idx += 2
                else:
                    new_pre_token.append(pre_token[idx])
                    idx += 1

            # 使用更新后的pre token替代
            new_pre_token = tuple(new_pre_token)
            pre_tokens_count[new_pre_token] += count

            # 使用新pre token更新pair记录
            for pair in zip(new_pre_token, new_pre_token[1:]):
                pairs_count[pair] += count
                pairs_in_pre_token[pair].add(new_pre_token)

    logging.info("merge finished")
    logging.info("-" * 40)

    return (vocab, merges)
