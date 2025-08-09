
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List


'''
A @dataclass in Python is a decorator that automatically generates boilerplate code for classes that are primarily used to store data — such as constructors (__init__), equality comparisons (__eq__), and nice string representations (__repr__).

It was introduced in Python 3.7 via the dataclasses module.
'''
@dataclass
class BPETokenizerParams:
    vocab: Dict[int, bytes]
    merges: Dict[Tuple[int, int], int]

def merge(indices: List[int], pair: Tuple[int, int], new_index: int) -> List[int]:
    """Replace all occurrences of a pair with a new merged index."""
    i = 0
    result = []
    while i < len(indices):
        if i < len(indices) - 1 and (indices[i], indices[i+1]) == pair:
            result.append(new_index)
            i += 2
        else:
            result.append(indices[i])
            i += 1
    return result

def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    '''
    A greedy way to do Byte Pair Encoding.
    Treat frequence subsequences as Token
    Merge High Frequence Token!
    '''
    # Start with the list of bytes from the string
    indices = list(map(int, string.encode("utf-8")))

    # Starting in 3.7, dicts will be guaranteed to retain insertion order.
    merges: Dict[Tuple[int, int], int] = {} # Must be ordered struct -> BSTMap
    vocab: Dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # initial vocab of bytes

    # Each time we merge **two adjacent (might alread merged) tokens**
    # This is a greedy algo
    for i in range(num_merges):
        # Count the number of occurrences of each adjacent pair
        counts = defaultdict(int)

        # compare adjacent tokens (remaining in indices is not used)
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1

        if not counts:
            break

        # Find the most frequent pair
        pair = max(counts, key=counts.get)
        index1, index2 = pair

        # Merge that pair into a new symbol
        new_index = 256 + i
        merges[pair] = new_index

        # Map new_index to the merged byte sequence -> Expand the Vocabulary
        vocab[new_index] = vocab[index1] + vocab[index2]

        # Replace all occurrences of the pair in the token stream -> shrink the indices
        indices = merge(indices, pair, new_index)

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    string = "Tokenizer: text to token index"
    num_merges = 10
    bpe_params = train_bpe(string, num_merges)
    print("Vocabulary:")
    for index, token in bpe_params.vocab.items():
        print(f"{index}: {repr(token)}")
    print("\nMerges:")

    for pair, new_index in bpe_params.merges.items():
        print(f"{pair} -> {new_index}")
# This code implements a simple Byte Pair Encoding (BPE) tokenizer.
