import argparse
import logging
import os

import numpy as np
import yaml

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.utils import setup_logging

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
chunk_size = config.get("chunk_size", 1e9)

logging.info(
    f"Initiate BPE tokenizer with vocab_filepath: '{vocab_filepath}', "
    f"merges_filepath: {merges_filepath}, special_tokens: {special_tokens}. "
    f"It's used to tokenize data: {data_path}. "
    f"Encoded token ids are saved to: {ids_save_path}"
)

tokenizer = BPETokenizer.from_files(
    vocab_filepath=vocab_filepath,
    merges_filepath=merges_filepath,
    special_tokens=special_tokens,
)

logging.info("BPE tokenizer initiated")

temp_file = ids_save_path.replace(".npy", ".tmp")

cur_size = chunk_size
ids_tmp = np.memmap(temp_file, dtype=np.uint16, mode="w+", shape=(cur_size,))

idx = 0
with open(data_path) as f:
    for token_id in tokenizer.encode_iterable(f):
        if idx >= cur_size:
            ids_tmp.flush()  # flush from memory to disk
            del ids_tmp
            cur_size += chunk_size
            # create a larger memmap file
            ids_tmp = np.memmap(
                temp_file, dtype=np.uint16, mode="r+", shape=(cur_size,)
            )
            logging.info(
                f"{idx} token ids processed and saved to disk, "
                f"expanded to {cur_size} tokens"
            )

        ids_tmp[idx] = token_id
        idx += 1

logging.info(f"Total {idx} token ids processed")

ids_tmp.flush()
del ids_tmp
final_ids = np.memmap(temp_file, dtype=np.uint16, mode="r", shape=(idx,))
np.save(ids_save_path, final_ids)
os.remove(temp_file)

logging.info("token ids are serialized")
