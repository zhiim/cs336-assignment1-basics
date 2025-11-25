import argparse
import logging
import os
from datetime import datetime

import yaml

from cs336_basics.train import train

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

parser = argparse.ArgumentParser(prog="Transformer LM trainer")
parser.add_argument("-c", "--config", required=True)
parser.add_argument("-r", "--resume", required=True, type=bool, default=False)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

train(config, args.resume)
