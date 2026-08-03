"""Test the MSL translator: FR -> MSL -> EN and EN -> MSL -> FR.

Encodes a text in one language, generates text in the other language from the
packets. Shows actual translations so we can judge quality by eye.

Usage:
    python -u -m msl.eval.test_translation --codec runs/translator_phase2_0.pt
"""

from __future__ import annotations

import argparse

import torch

from msl.data.ms1 import MS1, TextRenderer
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec import build_codec
from msl.utils.seeding import default_device, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="runs/translator_phase2_0.pt")
    ap.add_argument("--n", type=int, default=5, help="number of examples to show")
    args = ap.parse_args()

    seed_everything(0)
    device = default_device()
    tok = default_tokenizer()

    ckpt = torch.load(args.codec, map_location=device, weights_only=False)
    codec = build_codec(ckpt["cfg"], len(tok.itos)).to(device)
    codec.load_state_dict(ckpt["codec"])
    codec.eval()
    print(f"loaded {args.codec}")

    # Generate fresh states for testing (unseen).
    gen = MS1(min_k=4, max_k=16)
    renderer = TextRenderer()

    print("\n" + "=" * 80)
    print("TRADUCTION FR -> MSL -> EN")
    print("=" * 80)

    for i in range(args.n):
        s = gen.generate(seed=2_000_100 + i, k=8)
        fr_text = renderer.render(s, 0)  # FR view 0
        en_text = renderer.render(s, 5)  # EN view 5

        # Encode FR -> packets
        fr_ids = torch.tensor([tok.encode(fr_text, True, True)], dtype=torch.long, device=device)
        fr_mask = (fr_ids == tok.pad_id).long()
        with torch.no_grad():
            codes, z_q, _, _ = codec.encode_quantize(fr_ids, fr_mask)

        # Generate EN from packets
        lang_en = torch.tensor([1], device=device)  # 1 = EN
        with torch.no_grad():
            out_ids = codec.decoder.generate(z_q, lang_en, tok.bos_id, tok.eos_id, max_len=96)
        en_generated = tok.decode(out_ids[0].tolist())

        print(f"\n--- Exemple {i+1} (k={s.difficulty}) ---")
        print(f"  FR original:    {fr_text[:120]}")
        print(f"  EN reference:   {en_text[:120]}")
        print(f"  EN via MSL:     {en_generated[:120]}")

    print("\n" + "=" * 80)
    print("TRADUCTION EN -> MSL -> FR")
    print("=" * 80)

    for i in range(args.n):
        s = gen.generate(seed=2_000_200 + i, k=8)
        fr_text = renderer.render(s, 0)
        en_text = renderer.render(s, 5)

        # Encode EN -> packets
        en_ids = torch.tensor([tok.encode(en_text, True, True)], dtype=torch.long, device=device)
        en_mask = (en_ids == tok.pad_id).long()
        with torch.no_grad():
            codes, z_q, _, _ = codec.encode_quantize(en_ids, en_mask)

        # Generate FR from packets
        lang_fr = torch.tensor([0], device=device)  # 0 = FR
        with torch.no_grad():
            out_ids = codec.decoder.generate(z_q, lang_fr, tok.bos_id, tok.eos_id, max_len=96)
        fr_generated = tok.decode(out_ids[0].tolist())

        print(f"\n--- Exemple {i+1} (k={s.difficulty}) ---")
        print(f"  EN original:    {en_text[:120]}")
        print(f"  FR reference:   {fr_text[:120]}")
        print(f"  FR via MSL:      {fr_generated[:120]}")

    # Also show a round-trip: FR -> MSL -> FR
    print("\n" + "=" * 80)
    print("ALLER-RETOUR FR -> MSL -> FR")
    print("=" * 80)

    for i in range(3):
        s = gen.generate(seed=2_000_300 + i, k=8)
        fr_text = renderer.render(s, 0)

        fr_ids = torch.tensor([tok.encode(fr_text, True, True)], dtype=torch.long, device=device)
        fr_mask = (fr_ids == tok.pad_id).long()
        with torch.no_grad():
            codes, z_q, _, _ = codec.encode_quantize(fr_ids, fr_mask)
        lang_fr = torch.tensor([0], device=device)
        with torch.no_grad():
            out_ids = codec.decoder.generate(z_q, lang_fr, tok.bos_id, tok.eos_id, max_len=96)
        fr_generated = tok.decode(out_ids[0].tolist())

        print(f"\n--- Exemple {i+1} ---")
        print(f"  FR original:    {fr_text[:120]}")
        print(f"  FR via MSL:     {fr_generated[:120]}")


if __name__ == "__main__":
    main()
