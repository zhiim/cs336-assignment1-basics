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
parser.add_argument("-r", "--resume", required=True, type=bool, default=False)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

train(config, args.resume)
