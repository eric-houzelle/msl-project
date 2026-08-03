"""Test the MSL translator v2: FR -> MSL -> EN and EN -> MSL -> FR.

Uses top-k sampling instead of greedy to avoid repetition.
"""

from __future__ import annotations

import argparse

import torch

from msl.data.ms1 import MS1, TextRenderer
from msl.models.tokenizer import default_tokenizer
from msl.train.train_codec_v2 import build_codec_v2
from msl.utils.seeding import default_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="runs/v2_p2_translate_0.pt")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    seed_everything(42)
    device = default_device()
    tok = default_tokenizer()

    ckpt = torch.load(args.codec, map_location=device, weights_only=False)
    codec = build_codec_v2(ckpt["cfg"], len(tok.itos)).to(device)
    codec.load_state_dict(ckpt["codec"])
    codec.eval()
    print(f"loaded {args.codec} ({ckpt['n_params']:,} params)")

    gen = MS1(min_k=4, max_k=12)
    renderer = TextRenderer()

    def show_translation(src_text, src_lang_name, tgt_lang_id, tgt_lang_name, ref_text):
        src_ids = torch.tensor([tok.encode(src_text, True, True)], dtype=torch.long, device=device)
        src_mask = (src_ids == tok.pad_id).long()
        with torch.no_grad():
            codes, z_q, _, _ = codec.encode_quantize(src_ids, src_mask)
            lang_id = torch.tensor([tgt_lang_id], device=device)
            out_ids = codec.decoder.generate(z_q, lang_id, tok.bos_id, tok.eos_id,
                                             max_len=80, top_k=5, temperature=0.8)
        generated = tok.decode(out_ids[0].tolist())
        return generated

    print("\n" + "=" * 80)
    print("TRADUCTION FR -> MSL -> EN (top-k=5, temp=0.8)")
    print("=" * 80)
    for i in range(args.n):
        s = gen.generate(seed=2_000_100 + i, k=8)
        fr = renderer.render(s, 0)
        en_ref = renderer.render(s, 5)
        en_gen = show_translation(fr, "FR", 1, "EN", en_ref)
        print(f"\n--- Exemple {i+1} ---")
        print(f"  FR source:  {fr[:150]}")
        print(f"  EN ref:     {en_ref[:150]}")
        print(f"  EN MSL:     {en_gen[:150]}")

    print("\n" + "=" * 80)
    print("TRADUCTION EN -> MSL -> FR")
    print("=" * 80)
    for i in range(args.n):
        s = gen.generate(seed=2_000_200 + i, k=8)
        en = renderer.render(s, 5)
        fr_ref = renderer.render(s, 0)
        fr_gen = show_translation(en, "EN", 0, "FR", fr_ref)
        print(f"\n--- Exemple {i+1} ---")
        print(f"  EN source:  {en[:150]}")
        print(f"  FR ref:     {fr_ref[:150]}")
        print(f"  FR MSL:     {fr_gen[:150]}")

    print("\n" + "=" * 80)
    print("ALLER-RETOUR FR -> MSL -> FR")
    print("=" * 80)
    for i in range(3):
        s = gen.generate(seed=2_000_300 + i, k=8)
        fr = renderer.render(s, 0)
        fr_gen = show_translation(fr, "FR", 0, "FR", fr)
        print(f"\n--- Exemple {i+1} ---")
        print(f"  FR source:  {fr[:150]}")
        print(f"  FR MSL:     {fr_gen[:150]}")


if __name__ == "__main__":
    main()
