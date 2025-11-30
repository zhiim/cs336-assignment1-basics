import argparse
import logging
import os
from multiprocessing import Pool

import numpy as np
import yaml

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.train_bpe import find_chunk_boundaries
from cs336_basics.utils import setup_logging


def tokenize(args):
    file_path, start, end, tokenizer = args

    with open(file_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8")

        token_ids = tokenizer.encode(chunk)

    logging.info(
        f"tokenization in chunk: {start}-{end - 1} bytes finished. "
        f"{len(token_ids)} token ids generated"
    )

    return token_ids


script_name = os.path.basename(__file__)
script_name = os.path.splitext(script_name)[0]
setup_logging(script_name)


parser = argparse.ArgumentParser(prog="Tokenization using BPE tokenizer")
parser.add_argument("-c", "--config", required=True)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

vocab_filepath = config["vocab_filepath"]
merges_filepath = config["merges_filepath"]
special_tokens = config["special_tokens"]
data_path = config["data_path"]
ids_save_path = config["ids_save_path"]
num_chunk = config.get("num_chunk", 100)

logging.info(
    f"Initiate BPE tokenizer with vocab_filepath: '{vocab_filepath}', "
    f"merges_filepath: {merges_filepath}, special_tokens: {special_tokens}. "
    f"It's used to tokenize data: {data_path}. "
    f"Encoded token ids are saved to: {ids_save_path}, "
    f"with {num_chunk} chunks parallelly processed."
)

tokenizer = BPETokenizer.from_files(
    vocab_filepath=vocab_filepath,
    merges_filepath=merges_filepath,
    special_tokens=special_tokens,
)

logging.info("BPE tokenizer initiated")

with open(data_path, "rb") as f:
    boundaries = find_chunk_boundaries(
        f, num_chunk, special_tokens[0].encode("utf-8")
    )

logging.info(f"Chunk boundaries found: {boundaries}")

args_list = [
    (data_path, start, end, tokenizer)
    for start, end in zip(boundaries[:-1], boundaries[1:])
]
with Pool() as pool:
    results = pool.map(tokenize, args_list)

total_size = sum(len(result) for result in results)
out_file = np.memmap(
    ids_save_path, dtype=np.uint16, mode="w+", shape=(total_size,)
)

offset = 0
for result in results:
    out_file[offset : offset + len(result)] = result
    offset += len(result)
out_file.flush()
del out_file

logging.info("token ids are serialized")
