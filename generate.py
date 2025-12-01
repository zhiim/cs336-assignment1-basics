import argparse

import torch
import yaml

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.modules import RotaryPositionalEmbedding, Transformer
from cs336_basics.utils import decode

parser = argparse.ArgumentParser(
    prog="LM that generates response using user prompt"
)
parser.add_argument("-c", "--config", required=True)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

prompt = config["prompt"]
max_num_tokens = config["max_num_tokens"]
temperature = config["temperature"]
top_k = config["top_k"]
weight_path = config["weight_path"]
vocab_filepath = config["vocab_filepath"]
merges_filepath = config["merges_filepath"]
context_length = config["context_length"]
num_layers = config["num_layers"]
num_heads = config["num_heads"]
d_ff = config["d_ff"]
d_model = config["d_model"]
device = config["device"]
theta = config["theta"]

tokenizer = BPETokenizer.from_files(
    vocab_filepath=vocab_filepath,
    merges_filepath=merges_filepath,
    special_tokens=["<|endoftext|>"],
)

model = Transformer(
    vocab_size=len(tokenizer.vocab),
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    device=torch.device(device),
)
model.load_state_dict(
    torch.load(weight_path, map_location=torch.device(device))["model"]
)

rope = RotaryPositionalEmbedding(
    theta=theta,
    d_k=d_model // num_heads,
    max_seq_len=context_length,
    device=torch.device(device),
)

token_ids = tokenizer.encode(prompt)

input_tensor = (
    torch.Tensor(token_ids).unsqueeze(0).to(torch.int).to(torch.device(device))
)

result = decode(
    prompt=input_tensor,
    model=model,
    rope=rope,
    max_num_tokens=max_num_tokens,
    temperature=temperature,
    top_p=top_k,
    vocab=tokenizer.vocab,
)

print(prompt + result)
