import argparse
import logging
import os
import pickle
from datetime import datetime

import yaml

from cs336_basics.train_bpe import train_bpe

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

parser = argparse.ArgumentParser(prog="BPE tokenizer trainer")
parser.add_argument("-c", "--config", required=True)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

input_path = config["input_path"]
vocab_size = config["vocab_size"]
special_tokens = config["special_tokens"]
num_processes = config["num_processes"]

logger.info(
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

logger.info("vocab saved")

with open("merges.pkl", "wb") as f:
    pickle.dump(merges, f)

logger.info("merges saved")
