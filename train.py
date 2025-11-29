import argparse
import os

import yaml

from cs336_basics.train import train
from cs336_basics.utils import setup_logging

file_name = os.path.basename(__file__)
file_name = os.path.splitext(file_name)[0]
setup_logging(file_name)

parser = argparse.ArgumentParser(prog="Transformer LM trainer")
parser.add_argument("-c", "--config", required=True)
parser.add_argument("-r", "--resume_path", type=str, default=None)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

train(config, args.resume_path)
