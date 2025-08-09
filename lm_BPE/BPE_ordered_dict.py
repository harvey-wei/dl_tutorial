from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    @abstractmethod
    def encode(self, string: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, indices: list[int]) -> str:
        pass


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: Dict[int, bytes]     # index -> bytes
    merges: "OrderedDict[Tuple[int, int], int]"  # ordered merges


class CharacterTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        return list(map(int, string_bytes))

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        return string_bytes.decode("utf-8")


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and (indices[i], indices[i + 1]) == pair:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        for pair, new_index in self.params.merges.items():
            # This shorten the number of final tokens
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        #  dict.get(key, default=None) is a safe way to access dictionary values
        bytes_list = list(map(self.params.vocab.get, indices))
        if None in bytes_list:
            raise ValueError("Unknown token index during decoding.")
        # Joint list of byte obj into one byte obj and decode to UTF-8
        return b"".join(bytes_list).decode("utf-8")


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    indices = list(map(int, string.encode("utf-8")))
    merges: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
    vocab: Dict[int, bytes] = {x: bytes([x]) for x in range(256)} # basic vacab for each byte ASCII

    # We can also terminate according the size of vacab
    for i in range(num_merges):
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1

        if not counts:
            break

        # key=counts.get means counts.get[key] which is count
        # key in counts is pair of merged index
        '''
        max_pair = None
        max_count = -1
        for pair, count in counts.items():
            if count > max_count:
                max_pair = pair
                max_count = count
        '''
        pair = max(counts, key=counts.get)

        index1, index2 = pair
        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]  # expand
        indices = merge(indices, pair, new_index) # shrink

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    # Demo
    string = "language model language modeling"
    bpe_params = train_bpe(string, num_merges=10)
    tokenizer = BPETokenizer(bpe_params)

    encoded = tokenizer.encode(string)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    print("\nMerges:")
    for pair, idx in bpe_params.merges.items():
        print(f"{pair} → {idx}: {bpe_params.vocab[idx]!r}")
