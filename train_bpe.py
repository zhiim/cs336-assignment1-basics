import os
from collections import defaultdict, deque, Counter
from multiprocessing import Pool
import regex as re
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
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
        vocab_size: A positive integer that defines the maximum final vocabulary size.
        special_tokens: A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training

    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """
    # -- 1. initialize vocabulary ------------------------------------------

    # all 256 possible byte values
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_reverse = {bytes([i]): i for i in range(256)}

    # and special tokens
    new_ids = len(vocab)
    for token in special_tokens:
        if token not in vocab_reverse:
            vocab[new_ids] = token.encode("utf-8")
            vocab_reverse[token.encode("utf-8")] = new_ids
            new_ids += 1

    # -- 2. pre-tokenization -----------------------------------------------

    pre_tokens_count = Counter()  # 用于存储pre token和它们对于的计数

    special_tokens = [re.escape(token) for token in special_tokens]
    split_pattern = "|".join(special_tokens)

    # divide the whole text file into chunks
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # read chunk one by one
    args_list = [(input_path, split_pattern, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool() as pool:
        results = pool.map(pre_tokenization, args_list)
    for result in results:
        pre_tokens_count.update(result)

    # -- 3. BPE merges -----------------------------------------------------

    merges = []

    # vocab size达到要求就结束
    for new_id in range(len(vocab), vocab_size):
        pairs_count = defaultdict(int)

        # 对于每个pre token的每两个相邻byte，计算出现次数
        for pre_token, count in pre_tokens_count.items():
            for idx1, idx2 in zip(pre_token, pre_token[1:]):
                pairs_count[(idx1, idx2)] += count

        # 找到出现次数最多的pair
        merge_pair = max(pairs_count, key=pairs_count.get)

        # 添加到vocab中
        new_bytes = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        vocab[new_id] = new_bytes
        vocab_reverse[new_bytes] = new_id

        merges.append((vocab[merge_pair[0]], vocab[merge_pair[1]]))

        # 将新merge更新到pre token
        new_pre_tokens_count = Counter()
        for pre_token, count in pre_tokens_count.items():
            dq = deque(pre_token)
            new_token = []

            while dq:
                cur = dq.popleft()
                # 如果找到了pre token中和merge_pair相同的
                if len(dq) >= 1 and (cur, dq[0]) == merge_pair:
                    new_token.append(new_id)
                    dq.popleft()
                else:
                    new_token.append(cur)

            new_pre_tokens_count[tuple(new_token)] += count
        pre_tokens_count = new_pre_tokens_count

    return (vocab, merges)


if __name__ == "__main__":
    vocab, merges = train_bpe("./data/test.txt", vocab_size=300)
    print(vocab)
    print(merges)
