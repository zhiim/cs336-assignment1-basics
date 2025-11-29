import argparse
import logging
import os
import pickle

import yaml

from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils import setup_logging

file_name = os.path.basename(__file__)
file_name = os.path.splitext(file_name)[0]
setup_logging(file_name)

parser = argparse.ArgumentParser(prog="BPE tokenizer trainer")
parser.add_argument("-c", "--config", required=True)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

input_path = config["input_path"]
vocab_size = config["vocab_size"]
special_tokens = config["special_tokens"]
num_processes = config["num_processes"]

logging.info(
    f"training BPE tokenizer with data: '{input_path}', "
    f"vocab_size: {vocab_size}, special_tokens: {special_tokens}, "
    f"num_processes: {num_processes}"
)

vocab, merges = train_bpe(
    input_path=config["input_path"],
    vocab_size=config["vocab_size"],
    special_tokens=config["special_tokens"],
    num_processes=config["num_processes"],
)

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

logging.info("vocab saved")

with open("merges.pkl", "wb") as f:
    pickle.dump(merges, f)

logging.info("merges saved")
