# MSL — Machine Semantic Language

A learned, language-agnostic semantic codec that compresses meaning into discrete packets. Trained jointly with a decoder, it enables LLMs to think in MSL natively — 6× faster inference, 10% less memory, with faithful round-trip on real text.

## What is MSL?

MSL (Machine Semantic Language) is a research project to learn a bidirectional translator between human language and a native machine language made of discrete "semantic packets." The goal: train LLMs that reason in MSL directly, achieving the same semantic quality with measurably less compute, memory, and bandwidth.

Instead of tokenizing text into words, MSL compresses the *meaning* of a sentence into a short sequence of discrete codes (packets). A Transformer trained only on these packets can think, remember, and communicate without ever seeing human text.

## Key results (validated on prototype)

| Hypothesis | Question | Result | Status |
|---|---|---|---|
| H1 | Do packets carry meaning? | Gap +16 (synthetic), 4/5 faithful paraphrases (real text) | Validated |
| H2 | Is MSL an adoptable standard? | 87% adoption, 21% emergence | Validated |
| H3 | Is the economic gain real? | 6× faster end-to-end, 10% less memory | Validated |

## How it works

```
Text → Encoder (MiniLM) → Embedding → Quantizer (PQ) → MSL packets (384 bits)
                                                              ↓
Text ← GPT-2 (conditioned) ← z_q ←────────────────────────────┘
```

1. **Encoder**: `all-MiniLM-L6-v2` (pretrained, 22M params) converts a sentence to a 384-dim embedding.
2. **Quantizer**: Product Quantization (48 codebooks × 256 entries = 384 bits) compresses the embedding into discrete packets.
3. **Decoder**: GPT-2 (124M params) generates text conditioned on the quantized embedding via prefix embeddings.
4. **Native LLM**: A Transformer trained only on packet sequences — 6× faster than text, never sees human language.

## Project structure

```
msl_project/
├── Brief_projet_MSL_Kimi_K3.docx       # Original brief (source of truth)
├── docs/                                # 13 research documents
│   ├── 01_audit_critique.md             # Scientific audit, 5 hypotheses
│   ├── 02_experiences_falsification.md  # Falsification experiments
│   ├── 03_mvp.md                        # MVP spec (architecture, budget, go/no-go)
│   ├── 04-08_*.md                       # Results: H1, H3, economic verdict
│   ├── 09_traducteur_fonctionne.md      # 100% faithful round-trip (synthetic)
│   ├── 10-11_*.md                       # Real text: semantic capture + decoding
│   ├── 12_h2_standard_adoptable.md      # H2 validated: MSL is adoptable
│   ├── 13_installation_a10.md           # Debian 13 + A10 deployment guide
│   └── 14_plan_developpement.md          # Sequential plan: MiniLM → BGE-M3 → literal channel → scale
├── src/msl/                             # Source code
│   ├── data/                            # MS-1 generator, datasets, corpus builder
│   ├── models/                          # Encoder, quantizers (PQ/RVQ/FSQ), decoder, LLM
│   ├── train/                           # Training loops (codec, LLM, decoder)
│   └── eval/                            # Evaluation (H1 sweep, H2, round-trip)
├── tests/                               # Pytest suite (22 tests, 91% coverage)
├── configs/                             # YAML configs
└── AGENTS.md                            # Guide for AI coding tools
```

## Quick start

```bash
# Install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Train the codec on synthetic data (MS-1)
python -m msl.train.train_codec --config configs/mvp_smoke.yaml

# Test round-trip (100% faithful on synthetic)
python -c "
from msl.data.ms1 import MS1, TextRenderer
from msl.models.structured_codec import state_to_packets, packets_to_state
gen = MS1(min_k=4, max_k=12); r = TextRenderer()
s = gen.generate(seed=42, k=8)
p = state_to_packets(s); d = packets_to_state(p)
print('Original:', r.render(s, 0)[:100])
print('Round-trip:', r.render(d, 0)[:100])
"

# Train on real text (needs GPU)
python -u -m msl.data.download_hf --size 100000 --out runs/big_corpus.pt
python -u -m msl.train.train_text_decoder --steps 20000 --lr 3e-5
```

## Tech stack

- Python 3.11, PyTorch 2.13
- Transformers (GPT-2, all-MiniLM-L6-v2)
- Device: MPS (Apple Silicon) or CUDA (NVIDIA)
- Tests: pytest, ruff, mypy — all green

## Documentation

Read the docs in order:
1. `docs/01_audit_critique.md` — scientific audit and 5 hypotheses
2. `docs/03_mvp.md` — MVP specification
3. `docs/07_verdict_economique.md` — 6× speedup validated
4. `docs/12_h2_standard_adoptable.md` — MSL is an adoptable standard

## License

MIT

## Author

Eric Houzelle — eric.houzelle@gmail.com
