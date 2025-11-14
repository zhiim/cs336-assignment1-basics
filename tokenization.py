import argparse
import logging
import os
from datetime import datetime

import numpy as np
import yaml

from cs336_basics.bpe_tokenizer import BPETokenizer

# setting up logging
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8.8s] [%(filename)s:%(funcName)s] %(message)s"
)

file_name = os.path.basename(__file__)
file_name = os.path.splitext(file_name)[0]
file_handler = logging.FileHandler(
    f"{file_name}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log"
)
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)

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

logger.info(
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

ids = []
with open(data_path) as f:
    for token_id in tokenizer.encode_iterable(f):
        ids.append(token_id)

logging.info("all data tokenized")

ids_array = np.array(ids, dtype=np.uint16)
np.save(ids_save_path, ids_array)

logging.info("token ids are serialized")
