import os
import re
import pickle
import unicodedata
from collections import Counter, defaultdict


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer.
    - Base vocabulary consists of 256 bytes (UTF-8), guaranteeing no out-of-vocabulary tokens.
    - Automatic text normalization (unifying apostrophes, non-breaking spaces, control characters).
    - Uses regex pre-tokenization to keep words, punctuation, and whitespace distinct.
    - Fast inverted-index training algorithm (< 1 second on multi-megabyte texts).
    - Word-level chunk caching for near-instant encoding.
    """

    # Matches optional leading space + word, or punctuation, or whitespace
    SPLIT_PATTERN = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

    def __init__(self, vocab_size: int = 1000):
        self.target_vocab_size = vocab_size
        # Initial vocabulary: 256 individual bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def normalize(text: str) -> str:
        """Normalizes unicode, whitespace, apostrophes, and strips non-printable control characters."""
        # 1. Unicode NFC canonical composition
        text = unicodedata.normalize("NFC", text)
        # 2. Whitespace: convert non-breaking spaces, tabs, carriage returns
        text = text.replace("\xa0", " ").replace("\t", " ").replace("\r", "")
        # 3. Quotes & apostrophes: unify curly quotes/apostrophes with ASCII equivalents
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("“", '"').replace("”", '"')
        # 4. Standardize ellipsis and en-dash (keeps dialogue em-dash '—')
        text = text.replace("…", "...")
        text = text.replace("–", "-")
        # 5. Filter out non-printable control characters (except newline)
        text = "".join(
            ch for ch in text if ch == "\n" or not unicodedata.category(ch).startswith("C")
        )
        return text

    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        self.target_vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self._cache = {}

        # Normalize text before training
        text = self.normalize(text)

        # 1. Pre-tokenize text into word chunks and count their frequencies
        chunks = self.SPLIT_PATTERN.findall(text)
        chunk_counts = Counter(chunks)

        word_list = list(chunk_counts.keys())
        word_freqs = [chunk_counts[w] for w in word_list]
        splits = [list(w.encode("utf-8")) for w in word_list]

        # 2. Build initial adjacent pair frequencies and inverted index
        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)

        for w_idx, (toks, freq) in enumerate(zip(splits, word_freqs)):
            for j in range(len(toks) - 1):
                p = (toks[j], toks[j + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(w_idx)

        num_merges = vocab_size - 256
        if verbose:
            print(f"BPE: Training {num_merges} merges for vocab size {vocab_size}...")

        # 3. Iteratively merge the most frequent pair
        for i in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] <= 1:
                break

            new_idx = 256 + i
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            affected_words = list(pair_to_words[best_pair])
            del pair_counts[best_pair]
            del pair_to_words[best_pair]

            # Update only words containing best_pair
            for w_idx in affected_words:
                toks = splits[w_idx]
                freq = word_freqs[w_idx]

                # Decrement old adjacent pairs
                for j in range(len(toks) - 1):
                    p = (toks[j], toks[j + 1])
                    if p in pair_counts:
                        pair_counts[p] -= freq
                        if pair_counts[p] <= 0:
                            del pair_counts[p]
                    pair_to_words[p].discard(w_idx)

                # Merge occurrences of best_pair
                new_toks = []
                j = 0
                while j < len(toks):
                    if j < len(toks) - 1 and (toks[j], toks[j + 1]) == best_pair:
                        new_toks.append(new_idx)
                        j += 2
                    else:
                        new_toks.append(toks[j])
                        j += 1
                splits[w_idx] = new_toks

                # Increment new adjacent pairs
                for j in range(len(new_toks) - 1):
                    p = (new_toks[j], new_toks[j + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(w_idx)

        if verbose:
            print(f"BPE: Training complete! Final vocabulary size: {len(self.vocab)}")

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]

        toks = list(chunk.encode("utf-8"))
        while len(toks) >= 2:
            pairs = [(toks[j], toks[j + 1]) for j in range(len(toks) - 1)]
            # Find the pair that was merged earliest
            pair_to_merge = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair_to_merge not in self.merges:
                break
            new_id = self.merges[pair_to_merge]
            new_toks = []
            j = 0
            while j < len(toks):
                if j < len(toks) - 1 and (toks[j], toks[j + 1]) == pair_to_merge:
                    new_toks.append(new_id)
                    j += 2
                else:
                    new_toks.append(toks[j])
                    j += 1
            toks = new_toks

        self._cache[chunk] = toks
        return toks

    def encode(self, text: str) -> list[int]:
        """Encodes text into a list of token IDs."""
        text = self.normalize(text)
        tokens = []
        for chunk in self.SPLIT_PATTERN.findall(text):
            tokens.extend(self._encode_chunk(chunk))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of token IDs back into text."""
        raw_bytes = b"".join(self.vocab.get(t, b"") for t in tokens)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Saves tokenizer vocabulary and merges to file."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "merges": self.merges,
                    "target_vocab_size": self.target_vocab_size,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Loads a tokenizer from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        tok = cls(vocab_size=data["target_vocab_size"])
        tok.vocab = data["vocab"]
        tok.merges = data["merges"]
        return tok

    @classmethod
    def create(cls, corpus: str, vocab_size: int) -> "BPETokenizer":
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, vocab_size=vocab_size)
        return tokenizer
