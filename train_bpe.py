def pre_tokenization():
    pass


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
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
    # ----------------------------------------------------------------------
    init_vocab = {}
    for i in range(256):
        init_vocab[i] = bytes([i])
