"""Simple whitespace tokenizer for MS-1 text.

MS-1 text comes from parametric templates with a small closed vocabulary, so a
word-level tokenizer is sufficient and fully deterministic (no BPE training).
The vocab is built by scanning a sample of rendered states, then frozen.
"""

from __future__ import annotations

import re
from functools import lru_cache

from msl.data.ms1 import (
    ACTIONS,
    ATTRIBUTE_KEYS,
    MODALITIES,
    MS1,
    RELATION_TYPES,
    TextRenderer,
)

PAD, BOS, EOS, UNK, MASK = "<pad>", "<bos>", "<eos>", "<unk>", "<mask>"
SPECIALS = [PAD, BOS, EOS, UNK, MASK]
# Answer vocabulary (decoder targets for tasks).
ANSWER_TOKENS = ["unknown", "yes", "no"] + [str(i) for i in range(8)]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Tokenizer:
    def __init__(self) -> None:
        self.itos: list[str] = list(SPECIALS) + list(ANSWER_TOKENS)
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]
        self.mask_id = self.stoi[MASK]

    def _add(self, tok: str) -> None:
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def build(self, n_states: int = 2000, max_k: int = 32, seed: int = 12345) -> Tokenizer:
        g = MS1(min_k=2, max_k=max_k)
        r = TextRenderer()
        import numpy as np
        rng = np.random.default_rng(seed)
        # Static vocabulary from MS-1 atoms.
        for a in list(ATTRIBUTE_KEYS) + list(RELATION_TYPES) + list(ACTIONS) + list(MODALITIES):
            for tok in tokenize(a):
                self._add(tok)
        # Template vocabulary from rendered views.
        for _ in range(n_states):
            s = g.generate(seed=int(rng.integers(0, 1_000_000)), k=int(rng.integers(2, max_k + 1)))
            for v in range(9):
                for tok in tokenize(r.render(s, v)):
                    self._add(tok)
        return self

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = [self.bos_id] if add_bos else []
        ids.extend(self.stoi.get(t, self.unk_id) for t in tokenize(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i not in {self.pad_id, self.bos_id})


@lru_cache(maxsize=1)
def default_tokenizer() -> Tokenizer:
    return Tokenizer().build()


__all__ = ["Tokenizer", "default_tokenizer", "tokenize", "PAD", "BOS", "EOS", "UNK"]
